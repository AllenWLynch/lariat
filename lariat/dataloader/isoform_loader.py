from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable
from dataclasses import replace
import numpy as np
from functools import partial, cache
from itertools import groupby
import json
from webdataset.compat import WebDataset
from webdataset.filters import reraise_exception, pipelinefilter
from lariat.database.data_model import RelativeTranscript, IsoformRecord
from lariat.database import IsoformDB


def _sample_encoder(options: Dict[str, int], x: Optional[str]) -> int:
    if x is None:
        return 0
    try:
        return options[x]
    except KeyError:
        raise ValueError(
            f"Unknown option: {x} - known options: {list(options.keys())} - make sure the `db_subset` you provided matches the other shards."
        )


@cache
def technology_encoder(vocab_file: str) -> Callable[[Optional[str]], int]:
    with open(vocab_file, "r") as f:
        technologies = json.load(f)["technologies"]
    tech_to_id = {
        tech: i + 1 for i, tech in enumerate(sorted(technologies))
    }  # Start from 1
    return partial(_sample_encoder, tech_to_id)  # Unknown technology mapped to 0


@cache
def species_encoder(vocab_file: str) -> Callable[[Optional[str]], int]:
    with open(vocab_file, "r") as f:
        refid_map = json.load(f)["reference_id_to_species"]
    species_to_id = {
        sp: i + 1 for i, sp in enumerate(sorted(set(refid_map.values())))
    }  # Start from 1
    refkey_to_species_id: Dict[str, int] = {
        ref: species_to_id[sp] for ref, sp in refid_map.items() if sp
    }
    return partial(_sample_encoder, refkey_to_species_id)  # Unknown species mapped to 0


def _estimate_fwdbwd_vram(
    *,
    encoder_seqlen: int,
    decoder_seqlen: int,
    num_layers: int,
    encoder_fraction: float,
    model_dim: int,
    feedforward_dim: int,
    batch_size: int = 1,
    num_attention_heads: int = 8,
    include_optimizer: bool = True,
    dtype_bytes: int = 4,  # float32
) -> float:
    """
    Comprehensive estimate of VRAM usage for transformer model training.

    Accounts for:
    - Self-attention and cross-attention matrices
    - Feedforward layer activations
    - Model parameters
    - Gradients
    - Optimizer states (Adam: momentum + variance)
    - Activation storage for backpropagation

    Args:
        encoder_seqlen: Encoder sequence length
        decoder_seqlen: Decoder sequence length
        num_layers: Total number of transformer layers
        encoder_fraction: Fraction of layers allocated to encoder
        model_dim: Model hidden dimension
        feedforward_dim: Feedforward layer dimension
        batch_size: Batch size for activation memory calculation
        num_attention_heads: Number of attention heads
        include_optimizer: Whether to include optimizer state memory
        dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)

    Returns:
        Estimated VRAM usage in gigabytes
    """
    head_dim = model_dim // num_attention_heads

    def self_attention_mem(seq_len: int, batch_size: int) -> float:
        # Attention matrix: (batch_size, num_heads, seq_len, seq_len)
        attention_matrix = batch_size * num_attention_heads * (seq_len**2) * dtype_bytes

        # Q, K, V projections and outputs: (batch_size, seq_len, model_dim) each
        qkv_activations = (
            4 * batch_size * seq_len * model_dim * dtype_bytes
        )  # Q, K, V, output

        # Intermediate attention scores and softmax activations
        attention_scores = (
            batch_size * num_attention_heads * seq_len * seq_len * dtype_bytes
        )

        return attention_matrix + qkv_activations + attention_scores

    def cross_attention_mem(
        enc_seq_len: int, dec_seq_len: int, batch_size: int
    ) -> float:
        # Cross-attention matrix: (batch_size, num_heads, dec_seq_len, enc_seq_len)
        cross_attention_matrix = (
            batch_size * num_attention_heads * dec_seq_len * enc_seq_len * dtype_bytes
        )

        # Q (from decoder), K, V (from encoder), output
        qkv_activations = (
            batch_size
            * (dec_seq_len + 2 * enc_seq_len + dec_seq_len)
            * model_dim
            * dtype_bytes
        )

        # Attention scores
        attention_scores = (
            batch_size * num_attention_heads * dec_seq_len * enc_seq_len * dtype_bytes
        )

        return cross_attention_matrix + qkv_activations + attention_scores

    def feedforward_mem(seq_len: int, batch_size: int) -> float:
        # Input projection: (batch_size, seq_len, feedforward_dim)
        input_proj = batch_size * seq_len * feedforward_dim * dtype_bytes

        # Output projection back to model_dim
        output_proj = batch_size * seq_len * model_dim * dtype_bytes

        # Activation function intermediate results
        activation_intermediate = batch_size * seq_len * feedforward_dim * dtype_bytes

        return input_proj + output_proj + activation_intermediate

    def layer_norm_mem(seq_len: int, batch_size: int) -> float:
        # Layer norm statistics and normalized outputs (typically 2 per layer)
        return 2 * batch_size * seq_len * model_dim * dtype_bytes

    def parameter_mem() -> float:
        """Memory for model parameters"""
        encoder_layers = int(num_layers * encoder_fraction)
        decoder_layers = int(num_layers * (1 - encoder_fraction))

        # Per encoder layer: self-attention + feedforward + layer norms
        encoder_params = encoder_layers * (
            4 * model_dim * model_dim  # Q, K, V, O projections
            + 2 * model_dim * feedforward_dim  # FFN input and output projections
            + 4 * model_dim  # Layer norm parameters (2 layer norms per layer)
        )

        # Per decoder layer: self-attention + cross-attention + feedforward + layer norms
        decoder_params = decoder_layers * (
            4 * model_dim * model_dim  # Self-attention Q, K, V, O
            + 4 * model_dim * model_dim  # Cross-attention Q, K, V, O
            + 2 * model_dim * feedforward_dim  # FFN
            + 6 * model_dim  # Layer norm parameters (3 layer norms per layer)
        )

        total_params = encoder_params + decoder_params
        return total_params * dtype_bytes

    # Calculate memory for each component
    encoder_layers = int(num_layers * encoder_fraction)
    decoder_layers = int(num_layers * (1 - encoder_fraction))

    # Activation memory (stored for backpropagation)
    encoder_activation_mem = encoder_layers * (
        self_attention_mem(encoder_seqlen, batch_size)
        + feedforward_mem(encoder_seqlen, batch_size)
        + layer_norm_mem(encoder_seqlen, batch_size)
    )

    decoder_activation_mem = decoder_layers * (
        self_attention_mem(decoder_seqlen, batch_size)
        + cross_attention_mem(encoder_seqlen, decoder_seqlen, batch_size)
        + feedforward_mem(decoder_seqlen, batch_size)
        + layer_norm_mem(decoder_seqlen, batch_size)
    )

    # Model parameters
    param_mem = parameter_mem()

    # Gradients (same size as parameters)
    gradient_mem = param_mem

    # Optimizer states (Adam stores momentum and variance)
    optimizer_mem = 2 * param_mem if include_optimizer else 0

    # Input/output embeddings and other overhead (rough estimate)
    embedding_mem = 2 * model_dim * 50000 * dtype_bytes  # Rough vocab size estimate

    total_mem = (
        encoder_activation_mem
        + decoder_activation_mem
        + param_mem
        + gradient_mem
        + optimizer_mem
        + embedding_mem
    )

    return total_mem / 1e9  # Convert to gigabytes


def split_by_cost(
    transcripts: List[RelativeTranscript],
    *,
    max_cost: float,
    cost_fn: Callable[[RelativeTranscript], float],
) -> Iterable[List[RelativeTranscript]]:
    cummulative_cost = np.cumsum([cost_fn(t) for t in transcripts])
    for _, group in groupby(
        zip(transcripts, cummulative_cost), key=lambda x: int(x[1] // max_cost)
    ):
        grouped = [t for t, _ in group]
        yield grouped


def augment_sample(
    sample: IsoformRecord,
    context_dropout: float = 0.1,
    max_pad: int = 4096,
    random_state: Optional[np.random.RandomState] = None,
) -> IsoformRecord:
    return replace(
        sample, region=sample.region.slop_upstream(max_pad).slop_downstream(max_pad)
    )


def load_context_data(
    db: IsoformDB,
    vocab_file: str,
    record: IsoformRecord,
) -> Dict[str, Any]:
    return {
        "technology": technology_encoder(vocab_file)(record.technology_name),
        "species": species_encoder(vocab_file)(record.reference_id),
        "sequence": db.get_reference_sequence(record),
        "celltype_embeddings": {},
        "isoform_record": record,
    }


def unnest(
    nested_key: str,
    sample: Dict[str, Any],
) -> List[Dict[str, Any]]:
    repeat_items = {k: v for k, v in sample.items() if k != nested_key}
    return [{**repeat_items, nested_key: nested} for nested in sample[nested_key]]


def variable_size_batch(
    data,
    batch_cost: float,
    cost_fn: Callable[[RelativeTranscript], float],
    partial=True,
):
    batch = []
    current_cost = 0.0

    for sample in data:
        sample_cost = sum(cost_fn(t) for t in sample["transcripts"])
        if current_cost + sample_cost > batch_cost and len(batch) > 0:
            yield batch
            batch = []
            current_cost = 0.0
        batch.append(sample)
        current_cost += sample_cost

    if len(batch) == 0:
        return
    elif len(batch) > 0 and partial:
        yield batch


def get_weights(transcripts: List[RelativeTranscript]) -> List[float]: ...


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated = {}
    for key in batch[0].keys():
        collated[key] = [sample[key] for sample in batch]
    return collated


def get_dataloader(
    shards: str,
    isoform_db_path: str,
    vocab_file: str,
    sample_max_cost: float,
    batch_max_cost: float,
    cost_fn: Callable[[RelativeTranscript], float] = lambda x: 1,
    context_dropout: float = 0.1,
    max_pad: int = 4096,
) -> WebDataset:

    db = IsoformDB(isoform_db_path)

    augment_fn = partial(
        augment_sample,
        random_state=np.random.RandomState(42),
        context_dropout=context_dropout,
        max_pad=max_pad,
    )

    dataset = (
        WebDataset(shards)
        .shuffle(100)
        .decode()
        .map(
            lambda record: IsoformRecord.from_pyarrow_record(record["transcripts.msg"])
        )
        .map(augment_fn)  # No augmentation for now
        .map(partial(load_context_data, db, vocab_file))
        .map(
            lambda sample: {
                "transcripts": sample["isoform_record"].to_relative_transcripts(),
                **sample,
            }
        )
        .map_dict(
            transcripts=partial(
                split_by_cost, max_cost=sample_max_cost, cost_fn=cost_fn
            )
        )
        .map(partial(unnest, "transcripts"))
        .unlisted()
        .shuffle(1000)
        .map(
            lambda sample: {
                "uxid": list(map(lambda t: t.to_uxid(), sample["transcripts"])),
                "weights": get_weights(sample["transcripts"]),
                **sample,
            }
        )
        .compose(
            partial(variable_size_batch, batch_cost=batch_max_cost, cost_fn=cost_fn)
        )
        .map(collate)
    )

    return dataset
