#!/usr/bin/env python3
import sys
import pandas as pd

col_names = [
    'chrom', 'juncStart', 'juncEnd', 'juncName', 'juncScore', 'strand', 'spliceSite', 'acceptorsSkipped', 'exonsSkipped', 'donorsSkipped', 'anchor', 'knownDonor', 'knownAcceptor', 'knownJunction', # Columns from junctions annotate file
    'geneStart', 'geneEnd', 'geneName', 'geneID' # Columns from collpased annotated reference genome file
]

input_file = sys.stdin
output_file = sys.stdout

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

# Filter junctions
df = df[(df['juncScore'] > 5) | ((df['juncScore'] > 1) & (df['knownJunction'] == 1))]

# Calculate start and end offsets of junction based on gene start index
df['juncStartOff'] = df['juncStart'] - df['geneStart']
df['juncEndOff'] = df['juncEnd'] - 1 - df['geneStart'] # -1 b/c junctions annotate returns 1-based chromEnd

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