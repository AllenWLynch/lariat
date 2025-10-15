"""
The on-disk dataset for the track dataset loader is a zarr dset, which
is hierarchically organized by species, chromosome, and then chunked matrices
of shape (chunk_size, n_tracks) - chunk_size is typically 131072 and n_tracks
is the number of tracks in the dataset. The values are accessed like:
```
dset[species][chromosome][start:end, ...]
```
The input to the dataloader is a configuration with lists of dictionaries with the following keys:
{
  "species": str #species name to index database
  "intervals" : str #path to a BED file with intervals to load
  "fasta": str #path to a FASTA file with the genome sequence
}

In addition, the dataloader takes the `seqlen` parameter, which is the length of
the sequences to batch, the `batch_size` parameter, which is the number of sequences
to collate, and `buffer_size`, which is the number of `seqlen`-length chunks to
load into a memory buffer for random sampling.

To serve a batch, 
1. The dataloader randomly selects a species from the config - proportional
to the number of bases within the intervals for that species. Then, it randomly selects
a region from the intervals for that species.
2. It loads the corresponding track data and the genome sequence from the FASTA file.
3. It samples a random start position [0, seqlen - 1] within the region, and 
shatters the region into chunks of size `seqlen`, starting from that position.
4. It collects `buffer_size` chunks of size `seqlen` into a memory buffer.
5. It randomly samples `batch_size` chunks from the memory buffer and collates them
into a batch.
"""
from __future__ import annotations
from typing import Any, Iterable, List, Dict, Tuple, Generator, Annotated
from lariat.genome_utils import Region, stream_regions
from itertools import cycle, chain, starmap
from dataclasses import dataclass
from functools import partial
from pyfaidx import Fasta
import numpy as np
import torch
import zarr

__all__ = ["TrackDataset", "collate_tracks"]

@dataclass
class SpeciesConfig:
    name: str
    intervals: List[Region]
    fasta: Fasta

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpeciesConfig":
        return cls(
            name=d["species"],
            intervals=list(stream_regions(d["intervals"])),
            fasta=Fasta(d["fasta"])
        )
    

@dataclass
class Sample:
    track_data: np.ndarray  # shape (seqlen, n_tracks)
    seq: str                # length seqlen
    species: str
    region: Region

    def __len__(self) -> int:
        return self.track_data.shape[0]

    def __getitem__(self, idx: int | slice) -> Sample:
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return Sample(
            track_data=self.track_data[idx, :],
            seq=self.seq[idx],
            species=self.species,
            region=Region(
                chrom=self.region.chrom,
                start=self.region.start + (idx.start or 0),
                end=self.region.start + (idx.stop or self.track_data.shape[0])
            )
        )


class ShuffleBuffer:
    def __init__(
        self,
        it: Iterable, 
        max_size: int,
        random_state : np.random.Generator = np.random.default_rng()
    ):
        self.max_size = max_size
        self.iterator = iter(it)
        self.random_state = random_state
        self.buffer = []

    def __iter__(self):
        return self
    
    def __next__(self):
        while len(self.buffer) < self.max_size:
            self.buffer.append(next(self.iterator))
        if not self.buffer:
            raise StopIteration
        idx = self.random_state.choice(len(self.buffer))
        return self.buffer.pop(idx)


def _iter_shuffled(x: List[Any], random_state: np.random.Generator = np.random.default_rng()) -> Generator[Any, None, None]:
    """Infinitely yield elements of x in random order."""
    while True:
        indices = random_state.permutation(len(x))
        for i in indices:
            yield x[i]
    

def _iter_regions(
    configs: List[SpeciesConfig],
    random_state: np.random.Generator = np.random.default_rng()
) -> Generator[Tuple[SpeciesConfig, Region], None, None]:
    """
    Cycle through species, yielding (species_name, region) tuples.
    """
    species_iters = {
        config.name: _iter_shuffled(config.intervals, random_state)
        for config in configs
    }
    species_cycle = cycle([config for config in configs])
    for config in species_cycle:
        region = next(species_iters[config.name])
        yield (config, region)


def _load_region(
    species_config: SpeciesConfig,
    region: Region,
    *,
    dset: zarr.Group,
) -> Sample:
    """
    Load track data and genome sequence for a given species and region.
    """
    chrom = region.chrom
    start = region.start
    end = region.end

    # Load track data from zarr
    track_data: np.ndarray = dset[species_config.name][chrom][start:end, :] # type: ignore
    # Load genome sequence from FASTA
    fasta = species_config.fasta
    seq = fasta[chrom][start:end].seq.upper() # type: ignore

    return Sample(
        track_data=track_data,
        seq=seq,
        species=species_config.name,
        region=region
    )


def _chunk_region(
    sample: Sample,
    *,
    seqlen: int,
    random_state: np.random.Generator = np.random.default_rng()
) -> Generator[Sample, None, None]:
    """
    Given track data and sequence for a region, yield chunks of size `seqlen`.
    Start from a random offset within [0, seqlen - 1].
    """
    region_len = len(sample)
    if region_len < seqlen:
        return  # Region too short to yield any chunks
    start_offset = random_state.integers(0, seqlen)
    for start in range(start_offset, region_len - seqlen + 1, seqlen):
        end = start + seqlen
        yield sample[start:end]


def collate_tracks(
    batch: List[Sample]
) -> Tuple[
    Annotated[torch.Tensor, "batch seqlen n_tracks"],
    Annotated[torch.Tensor, "batch sequences"],
    Annotated[torch.Tensor, "cu_seqlens"]
]:
    ...


class TrackDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        dset: str,
        configs: List[Dict[str, Any]],
        *,
        seqlen: int = 8192,
        batch_size: int = 64,
        buffer_size: int = 1024,
        random_seed: int = 42
    ):
        self.dset = dset
        self.species_configs = [SpeciesConfig.from_dict(c) for c in configs]
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.random_state = np.random.default_rng(random_seed)

    def __iter__(self):
        zroot = zarr.open(self.dset, mode="r")
        regions = _iter_regions(self.species_configs, self.random_state)
        regions = starmap(partial(_load_region, dset=zroot), regions) # type: ignore
        chunks = map(partial(_chunk_region, seqlen=self.seqlen, random_state=self.random_state), regions)
        chunks = iter(chain.from_iterable(chunks))
        buffer = ShuffleBuffer(chunks, self.buffer_size, self.random_state)
        return buffer
