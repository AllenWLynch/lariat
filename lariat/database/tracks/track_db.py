from typing import List, Optional
import numpy as np
import pandas as pd
import zarr
from lariat.genome_utils import Region

def load_metadata(zarr_path: str) -> pd.DataFrame:
    zroot = zarr.open_group(zarr_path, mode="r")
    metadata = pd.DataFrame(dict(zroot.attrs)).set_index("accession")
    return metadata

def load_region(
    zarr_path: str,
    region: Region,
    accessions: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Load track data for a specific genomic region from a Zarr dataset.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    track_query : pd.DataFrame
        DataFrame with track metadata, indexed by accession.
    region : Region
        Genomic region to load (chrom, start, end).

    Returns
    -------
    np.ndarray
        Array of shape (region_length, n_tracks) with track data.
    """
    zroot = zarr.open_group(zarr_path, mode="r")
    if region.chrom not in zroot:
        raise ValueError(f"Chromosome {region.chrom} not found in Zarr dataset.")
    
    chrom_data = zroot[region.chrom]
    if accessions is None:
        return chrom_data[region.start:region.end, :].T # type: ignore
    
    db_accs = list(dict(zroot.attrs)["accession"]) # type: ignore
    indices = [db_accs.index(acc) for acc in accessions]

    return chrom_data[region.start:region.end, indices].T # type: ignore
