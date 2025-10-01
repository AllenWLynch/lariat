from typing import Callable, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

poisson_nll = partial(F.poisson_nll_loss, log_input=True, reduction="sum")

@torch.compile
def cu_logsumexp_Nd(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    x: [N, D] tensor, concatenation of D sequences.
    cu_seqlens: [B+1] tensor with cu_seqlens[0]==0, cu_seqlens[-1]==N.

    Returns:
      Tensor of shape [B, D] where out[b] = logsumexp(x[cu_seqlens[b]:cu_seqlens[b+1]], dim=0).
      Empty segments (length 0) return -inf.
    """
    # lengths per segment and number of segments
    lengths = cu_seqlens.diff()  # [B]
    B = lengths.numel()

    # Map each element in x to its segment id in [0, B)
    seg_ids = torch.repeat_interleave(
        torch.arange(B, device=x.device, dtype=torch.long), lengths.clamp_min(0)
    )  # [N]

    # 1) per-segment max for numerical stability
    m = torch.full((B, x.shape[1]), float("-inf"), device=x.device, dtype=x.dtype)
    # index_reduce_ is fused and fast on recent PyTorch
    m.index_reduce_(0, seg_ids, x, reduce="amax")  # m[b] = max of segment b

    # 2) sum of shifted exponentials
    # gather the max for each element, subtract, exp, then segment-sum
    shifted = torch.exp(x - m.index_select(0, seg_ids))
    s = torch.zeros_like(m)
    s.index_add_(0, seg_ids, shifted)  # s[b] = sum(exp(x_i - m[b]))

    # handle empty segments: s==0 => log(0) = -inf, m + (-inf) = -inf (desired)
    return m + torch.log(s)


class FactorizedPoissonLoss(nn.Module):
    """
    Separate the poisson nll into two terms: shape and rate.
    The shape term is analogous to a multinomial loss, where we predict
    the distribution of counts within each segment, independent of the total count.
    The rate term is a Poisson loss on the total count within each segment.
    """

    def __init__(
        self,
        dim: int,
        shape_factor: float = 1.0,
        rate_factor: float = 1.0,
        output_dim: int = 1,
        linear_cls: Callable[..., torch.nn.Module] = nn.Linear,
    ):
        super().__init__()
        self.shape_factor = shape_factor
        self.rate_factor = rate_factor
        self.output_dim = output_dim
        self.linear_project = linear_cls(dim, output_dim)  # project to log rate

    @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        target: torch.Tensor,
        cu_seqlens: torch.Tensor,
        inference_params: Optional[dict] = None,
        seq_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        hidden_states: [S, D] float tensor of sequence features
        target: [S, R] float tensor of counts
        cu_seqlens: [B+1] int tensor of segment boundaries into S
        inference_params: unused
        seq_idx: unused
        
        Returns:
          scalar loss
        """
        S = hidden_states.size(0)
        B = cu_seqlens.numel() - 1
        batch_id = (
            torch.searchsorted(
                cu_seqlens, torch.arange(S, device=hidden_states.device), right=True
            )
            - 1
        )

        preds = self.linear_project(hidden_states) # [S, R]
        rate_predictions = cu_logsumexp_Nd(preds, cu_seqlens)

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            preds = preds.float()
            rate_predictions = rate_predictions.float()
            target = target.float()

            with torch.no_grad():
                # Compute rate targets by summing target counts within each segment
                rate_targets = target.new_zeros((B, self.output_dim))
                rate_targets.index_add_(0, batch_id, target)  # [B, R]
                
                # Normalize targets to get shape (multinomial probabilities)
                # If rate_targets is zero, this will produce NaNs, which should be set to ones
                shape_target = target / rate_targets.index_select(0, batch_id)
                shape_target = shape_target.nan_to_num(1.0)

                deviance_loss = (
                    self.shape_factor * poisson_nll(shape_target, shape_target, log_input=False)
                    + self.rate_factor * poisson_nll(rate_targets, rate_targets, log_input=False)
                )

            shape_loss = poisson_nll(
                preds - rate_predictions.index_select(0, batch_id),
                shape_target,
            )
            rate_loss = poisson_nll(rate_predictions, rate_targets)

            model_loss = self.shape_factor * shape_loss + self.rate_factor * rate_loss
            return (model_loss - deviance_loss) / S
