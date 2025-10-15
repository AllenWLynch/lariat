#!/usr/bin/env python3
"""
Calculate junction annotation offsets relative to gene coordinates.

This script processes junction annotation data intersected with gene annotations,
filtering junctions by score thresholds and computing offsets relative to gene start positions.
"""
import sys
import argparse
import pandas as pd


def calculate_junction_offsets(
    input_file,
    output_file,
    min_score_known=1,
    min_score_unknown=5
):
    col_names = [
        'chrom', 'juncStart', 'juncEnd', 'juncName', 'juncScore', 'strand', 
        'spliceSite', 'acceptorsSkipped', 'exonsSkipped', 'donorsSkipped', 
        'anchor', 'knownDonor', 'knownAcceptor', 'knownJunction', 
        # Columns from junctions annotate file
        'geneStart', 'geneEnd', 'geneName', 'geneID' 
        # Columns from collapsed annotated reference genome file
    ]

    df = pd.read_csv(
        input_file, 
        sep='\t', 
        header=None, 
        names=col_names,
        dtype={
            'chrom': str,
            'juncStart': int,
            'juncEnd': int,
            'juncName': str,
            'juncScore': int,
            'strand': str,
            'spliceSite': str,
            'acceptorsSkipped': int,
            'exonsSkipped': int, 
            'donorsSkipped': int,
            'anchor': str, 
            'knownDonor': int, 
            'knownAcceptor': int, 
            'knownJunction': int,
            'geneStart': int,
            'geneEnd': int,
            'geneName': str,
            'geneID': str
        }
    )

    # Filter junctions based on score thresholds
    # Known junctions: juncScore > min_score_known AND knownJunction == 1
    # Unknown junctions: juncScore > min_score_unknown
    df = df[
        (df['juncScore'] > min_score_unknown) | 
        ((df['juncScore'] > min_score_known) & (df['knownJunction'] == 1))
    ]

    # Calculate start and end offsets of junction based on gene start index
    df['juncStartOff'] = df['juncStart'] - df['geneStart']
    df['juncEndOff'] = df['juncEnd'] - 1 - df['geneStart']  # -1 b/c junctions annotate returns 1-based chromEnd

    # Group junctions by gene and store junctions as list of tuples (start, end, score)
    res = (
        df.groupby(['chrom', 'geneName', 'geneID', 'strand', 'geneStart', 'geneEnd']) 
        .apply(
            lambda g: list(zip(g['juncStartOff'], g['juncEndOff'], g['juncScore'])),
            # include_group=True
        )
        .reset_index(name='junctions')
    )

    # Output file includes header with column names
    res.to_csv(output_file, sep='\t', index=False)
    
    return res


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Calculate junction annotation offsets relative to gene coordinates.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='Input TSV file with junction and gene annotations (default: stdin)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help='Output TSV file for junction offsets (default: stdout)'
    )
    
    parser.add_argument(
        '--min-score-known',
        type=int,
        default=1,
        help='Minimum score threshold for known junctions'
    )
    
    parser.add_argument(
        '--min-score-unknown',
        type=int,
        default=5,
        help='Minimum score threshold for unknown junctions'
    )
    
    args = parser.parse_args()
    
    try:
        calculate_junction_offsets(
            args.input,
            args.output,
            min_score_known=args.min_score_known,
            min_score_unknown=args.min_score_unknown
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Close files if they're not stdin/stdout
        if args.input != sys.stdin:
            args.input.close()
        if args.output != sys.stdout:
            args.output.close()


if __name__ == '__main__':
    main() 