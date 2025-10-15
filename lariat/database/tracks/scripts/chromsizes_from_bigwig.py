import pyBigWig
import sys
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(
        description="Extract chromosome sizes from a BigWig file."
    )
    parser.add_argument("bigwigs", nargs='+', help="Input BigWig file paths")
    args = parser.parse_args()

    if len(args.bigwigs) == 0:
        print("No BigWig files provided.", file=sys.stderr)
        sys.exit(1)

    get_chromsizes = lambda bw_path: pyBigWig.open(bw_path).chroms()
    chrom_sizes = list(map(get_chromsizes, args.bigwigs))

    # Get the union of all chromosomes across all BigWigs
    # and assert that sizes match where chromosomes overlap
    choose_sizes = defaultdict(set)
    for cs in chrom_sizes:
        for chrom, size in cs.items():
            choose_sizes[chrom].add(size)

    chrom_sizes = {}
    for chrom, sizes in choose_sizes.items():
        if len(sizes) > 1:
            print(f"Error: Chromosome {chrom} has differing sizes across BigWigs: {sizes}", file=sys.stderr)
            sys.exit(1)
        chrom_sizes[chrom] = sizes.pop()

    # Output chromosome sizes to stdout
    chrom_sizes = dict(sorted(chrom_sizes.items(), key=lambda item: item[0]))  # sort by chrom name
    for chrom, length in chrom_sizes.items():
        print(f"{chrom}\t{length}")

if __name__ == "__main__":
    main()