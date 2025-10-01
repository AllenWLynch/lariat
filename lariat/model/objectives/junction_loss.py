import torch
import torch.nn.functional as F
from typing import Tuple, Callable, Optional

torch._dynamo.config.capture_scalar_outputs = True


@torch.compile
def cu_logsumexp1d(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    x: 1D tensor of shape [total_len], concatenation of sequences.
    cu_seqlens: 1D tensor of shape [B+1] with cu_seqlens[0]==0, cu_seqlens[-1]==len(x).

    Returns:
      Tensor of shape [B], where out[b] = logsumexp(x[cu_seqlens[b]:cu_seqlens[b+1]]).
      Empty segments (length 0) return -inf.
    """
    # lengths per segment and number of segments
    lengths = cu_seqlens.diff()  # [B]
    B = lengths.numel()

    # Map each element in x to its segment id in [0, B)
    seg_ids = torch.repeat_interleave(
        torch.arange(B, device=x.device, dtype=torch.long), lengths.clamp_min(0)
    )  # [total_len]

    # 1) per-segment max for numerical stability
    m = torch.full((B,), float("-inf"), device=x.device, dtype=x.dtype)
    # index_reduce_ is fused and fast on recent PyTorch
    m.index_reduce_(0, seg_ids, x, reduce="amax")  # m[b] = max of segment b

    # 2) sum of shifted exponentials
    # gather the max for each element, subtract, exp, then segment-sum
    shifted = torch.exp(x - m.index_select(0, seg_ids))
    s = torch.zeros_like(m)
    s.index_add_(0, seg_ids, shifted)  # s[b] = sum(exp(x_i - m[b]))

    # handle empty segments: s==0 => log(0) = -inf, m + (-inf) = -inf (desired)
    return m + torch.log(s)


@torch.compile
def cu_sparse_softmax(
    logits: torch.Tensor,  # N
    indices: torch.Tensor,  # M
    weights: torch.Tensor,  # M
    cu_seqlens: torch.Tensor,  # B+1
) -> torch.Tensor:
    batch_ids = torch.searchsorted(cu_seqlens, indices, right=True) - 1  # [M]
    z = cu_logsumexp1d(logits, cu_seqlens).index_select(0, batch_ids)  # [M]
    return weights @ (logits.index_select(0, indices) - z)


@torch.compile
def _attention_lse(q, k, query_indices, cu_seqlens, *, pad_multiple=1):
    device = q.device
    M, D = q.shape
    N = k.shape[0]

    # per-query segment start and valid length
    seg_id = torch.searchsorted(cu_seqlens, query_indices, right=True) - 1  # [M]
    starts = cu_seqlens.index_select(0, seg_id)  # [M]
    ends = query_indices  # [M]
    lengths = (ends - starts).clamp_min(0)  # [M]

    # ---- sentinel padding ----
    # k_pad: [N+1, D], last row is zeros (or any value; it will be masked out anyway)
    k_pad = F.pad(k, (0, 0, 0, 1))  # pad one row at bottom
    sentinel = torch.tensor(N, device=device)  # index of the sentinel row

    Lmax = int(lengths.max().item())
    Lmax = ((Lmax + pad_multiple - 1) // pad_multiple) * pad_multiple

    cols = torch.arange(Lmax, device=device).unsqueeze(0)  # [1, Lmax]
    base = starts.unsqueeze(1) + cols  # [M, Lmax]
    mask = cols < lengths.unsqueeze(1)  # [M, Lmax]
    safe_base = torch.where(mask, base, sentinel)  # [M, Lmax] ∈ [0..N]

    # Gather K once: [M, Lmax, D]
    K_batch = k_pad.index_select(0, safe_base.reshape(-1)).view(M, Lmax, D)

    # One batched matmul: [M, Lmax]
    logits = torch.bmm(q.float().unsqueeze(1), K_batch.transpose(1, 2).float()).squeeze(
        1
    )
    logits = logits

    # Kill padded positions (blocks grads there)
    logits = logits.masked_fill(~mask, float("-inf"))

    return torch.logsumexp(logits, dim=-1).to(q.dtype)  # [M]


@torch.compile
def cu_joint_softmax(
    Q: torch.Tensor,  # [M, D]
    K: torch.Tensor,  # [N, D]
    marginal_logits: torch.Tensor,  # [N]
    cu_seqlens: torch.Tensor,  # [B+1] segment boundaries into K,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    weights: torch.Tensor,  # [M']
):
    # per query sum of weights
    marginal_i_idx = torch.unique(i_idx)
    M = marginal_i_idx.size(0)
    W_i = torch.zeros(M, device=Q.device, dtype=Q.dtype).index_add_(
        0, torch.searchsorted(marginal_i_idx, i_idx), weights
    )

    # logZ per query (i attends to its prefix within segment)
    logZ = _attention_lse(
        Q.index_select(0, marginal_i_idx),
        K,
        query_indices=marginal_i_idx,
        cu_seqlens=cu_seqlens,
        pad_multiple=128,
    )

    conditional_logits = (Q.index_select(0, i_idx) * K.index_select(0, j_idx)).sum(-1)
    conditional_term = (weights * conditional_logits).sum() - (W_i * logZ).sum()

    marginal_term = cu_sparse_softmax(marginal_logits, marginal_i_idx, W_i, cu_seqlens)
    return conditional_term, marginal_term


class JunctionLoss(torch.nn.Module):
    """
    This module expects the target data to be in the format of junction indices and weights,
    where the first tensor contains the indices of the acceptor sites (the 3' end of the junction w.r.t. the gene body)
    and the second tensor contains the donor sites (the 5' end of the junction w.r.t. the gene body).
    The third tensor contains the weights (counts or probabilities) associated with each junction.

    Thus, the acceptor indices should always be greater than the donor indices for valid junctions.
    """

    def __init__(
        self,
        dim: int,
        linear_cls: Callable[..., torch.nn.Module] = torch.nn.Linear,
        layernorm_cls: Callable[..., torch.nn.Module] = torch.nn.LayerNorm,
    ):
        super().__init__()
        self.linear_project = linear_cls(dim, 2 * dim + 1)  # project to 3*(Q,K,V)
        self.layer_norm = layernorm_cls(2 * dim + 1)
        self.dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,  # [L, D]
        *,
        target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # ([M'], [M']) indices into Q, K for nonzero weights,
        cu_seqlens: torch.Tensor,  # [B+1] segment boundaries into K
        inference_params: Optional[dict] = None,
        seq_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        QKL = self.linear_project(hidden_states)  # [L, 2*D + 1]

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            # Cast to float32 for the high dynamic range logit computations
            QKL = self.layer_norm(QKL)
            Q, K, logit = torch.split(QKL.float(), [self.dim, self.dim, 1], dim=-1)
            logit = logit.squeeze(-1)  # [L]

            c, m = cu_joint_softmax(Q, K, logit, cu_seqlens, *target)
            return -(c + m)
