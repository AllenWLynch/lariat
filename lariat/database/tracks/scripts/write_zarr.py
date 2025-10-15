import json
import argparse
import os
import sys
import zarr
import numpy as np
import pyBigWig
from tqdm import tqdm
import re
from typing import Dict, List, Tuple

def read_bw_chunk(bw_path, chrom, start, end):
    try:
        with pyBigWig.open(bw_path) as bw:
            vals = bw.values(chrom, start, end, numpy=True)
            vals = np.nan_to_num(vals)
        return vals
    except Exception as e:
        print(f"Error reading {bw_path}, {chrom}:{start}-{end}: {e}")
        return np.zeros(end - start)  # Return zeros on error


DTYPE=np.float16
TARGET_READ_DEPTH=1e8

def quantize(arr, read_depth, max_val) -> np.ndarray:
    """
    Normalize to the same read depth across files (100 million reads),
    then use max_val to scale to determine the least-lossy compression to float16.
    """    
    if read_depth <= 0:
        return np.zeros_like(arr, dtype=DTYPE)
    normalized_arr = arr * (TARGET_READ_DEPTH / read_depth)
    normalized_arr = np.clip(normalized_arr, 0, 65500)
    return normalized_arr.astype(DTYPE)

# ----------------------------
# MAIN
# ----------------------------
def write_quantized_zarr(
    records: List[Dict[str, str]],
    output_zarr: str,
    chunk_size: int = 131072,
    n_threads: int = 1,
    *,
    chromsizes: Dict[str, int],
):
    
    # pivot the records (list of dicts) into dict of lists
    print("Preparing metadata...", file=sys.stderr)
    metadata = {key: [r.get(key, "unknown") for r in records] for key in records[0]}
    bigwig_files = metadata.pop("file")
    n_files = len(bigwig_files)
    
    # use main chromosomes only
    chromosomes = [
        chrom for chrom in chromsizes.keys() 
        if re.match(r"^(chr)?([1-9][0-9]?)$", chrom)
    ]

    def _get_stats(bw_path: str) -> Tuple[float, float]:
        with pyBigWig.open(bw_path) as bw:
            header = bw.header()
        return header["maxVal"], header["sumData"]

    max_vals, read_depth = list(zip(*map(_get_stats, bigwig_files)))

    # Column metadata
    column_metadata = {
        **metadata,
        "max_val": max_vals,
        "read_depth": read_depth,
    }

    print("Creating dataset ...", file=sys.stderr)
    zroot = zarr.open_group(output_zarr, mode="w")
    # annotate the columns with metadata
    zroot.attrs.update(column_metadata)

    # ----------------------------
    # OPTIMIZED WRITE CHUNKS
    # ----------------------------
    chrom_pbar = tqdm(
        chromosomes, 
        desc="Processing chromosomes", position=0, leave=True, ncols=100, unit="chrom",
        bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}"
    )
    
    for chrom in chrom_pbar:
        
        length = chromsizes[chrom]
        
        if os.path.exists(os.path.join(output_zarr, chrom)):
            raise ValueError(f"Zarr array for chromosome {chrom} already exists in {output_zarr}. Please remove it before proceeding.")

        zds = zroot.create_array(
            chrom,
            shape=(length, n_files),
            chunks=(chunk_size, n_files), # minimum chunk size we'll load during training
            dtype=DTYPE
        )
        
        # Pre-allocate reusable buffer for the largest chunk
        max_chunk_len = min(chunk_size, length)
        chunk_buffer = np.zeros((max_chunk_len, len(bigwig_files)), dtype=DTYPE)
        chunk_buffer = np.asfortranarray(chunk_buffer) # make the memory layout column-major for faster writes

        # Create inner progress bar for chunks within this chromosome
        chunk_ranges = list(range(0, length, chunk_size))
        chunk_pbar = tqdm(
            total=len(chunk_ranges)*n_files, desc=f"Writing data", position=1, leave=False, ncols=100, unit="chunk",
            bar_format="{l_bar}{bar}|"
        )
        
        for start in chunk_ranges:
            end = min(start + chunk_size, length)
            chunk_len = end - start

            # Slice the pre-allocated buffer to avoid re-allocation
            chunk_data = chunk_buffer[:chunk_len, :]

            # Read BigWig files sequentially for this chunk
            for i, bw_path in enumerate(bigwig_files):
                raw_data = read_bw_chunk(bw_path, chrom, start, end)
                chunk_data[:, i] = quantize(raw_data, read_depth[i], max_vals[i])
                chunk_pbar.update(1)

            # Write chunk to Zarr and update progress
            zds[start:end, :] = chunk_data
        
        chunk_pbar.close()
    
    chrom_pbar.close()


def main():

    parser = argparse.ArgumentParser(
        description="Convert BigWigs to a chunked Zarr store with per-file sqrt-quantization."
    )
    parser.add_argument("json_file", help="JSON file describing BigWig files and metadata")
    parser.add_argument("output_zarr", help="Output Zarr store path")
    parser.add_argument("--chunk-size", type=int, default=131072, help="Genome chunk size (default 131072)")
    parser.add_argument("--chromsizes", required=True, help="Chromosome sizes file (two-column TSV)")
    parser.add_argument("--n-threads", "-@", type=int, default=1, help="Number of threads for parallel reading")
    args = parser.parse_args()

    with open(args.json_file) as f:
        records = json.load(f)

    # Read chromosome sizes from file
    chromsizes = {}
    with open(args.chromsizes) as f:
        for line in f:
            chrom, size = line.strip().split("\t")
            chromsizes[chrom] = int(size)

    write_quantized_zarr(
        records,
        args.output_zarr,
        chunk_size=args.chunk_size,
        n_threads=args.n_threads,
        chromsizes=chromsizes,
    )

if __name__ == "__main__":
    main()
