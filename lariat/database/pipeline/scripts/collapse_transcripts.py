#!/usr/bin/env python3
import sys
import pandas as pd

col_names = [
    'chrom', 'source', 'feature', 'chromStart', 'chromEnd', 'score', 'strand', 'phase', 'attributes'
]

input_file = sys.stdin # BED conversion of annotated GTF reference genome (comprehensive gene annotation, CHR)
output_file = sys.stdout

df = pd.read_csv(input_file, sep='\t', header=None, names=col_names
)

df = df.dropna(subset=['chromStart', 'chromEnd'])

df['chromStart'] = df['chromStart'].astype(int)
df['chromEnd'] = df['chromEnd'].astype(int)

df['geneID'] = df['attributes'].str.extract(r'geneID=([^;]+)')
df['geneName'] = df['attributes'].str.extract(r'gene_name=([^;]+)')

transcripts = df[df['feature'] == 'transcript']

# Group all transcripts by their associated gene
collapsed = transcripts.groupby(['chrom', 'geneName', 'geneID', 'strand']).agg(
    chromStart=('chromStart', 'min'),
    chromEnd=('chromEnd', 'max')
).reset_index()

collapsed['score'] = '.'

# Reorder columns to match BED6 format
collapsed = collapsed[['chrom', 'chromStart', 'chromEnd', 'geneName', 'score', 'strand', 'geneID']]

# Output file includes header with column names
collapsed.to_csv(output_file, sep='\t', index=False)