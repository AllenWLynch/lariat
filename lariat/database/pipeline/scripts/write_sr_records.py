from typing import Any, Dict, List, Iterable, Tuple, Optional
import pandas as pd
import ast
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from dataclasses import replace

from lariat.database import data_model
from lariat.genome_utils import Strand
from lariat.database.pipeline.config_models import QuantitationConfig
from lariat.database.pipeline.scripts.utils import (
    parse_intervals,
    logger,
    read_genes
)

def format_records(
        *,
        output_path: str,
        gene_annotation_file: str,
        bed_file: str,
        dataset_id: Optional[str] = None,
        reference_id: Optional[str] = None,
        is_single_cell: bool = False,
        celltypes: Optional[Dict[str, float]] = None,
        technology_name: Optional[str] = None
):
    def parse_junctions(df, gene_id):
        tuples_row = df.loc[df["geneID"] == gene_id]

        if len(tuples_row) == 0: # Some genes may not be present in BED file b/c CZ pipeline filtered for junctions w/ score>5
            return None 

        if len(tuples_row) != 1:
            raise ValueError(f"Expected exactly one row for geneID={gene_id}, but found {len(tuples_row)} rows.")

        try:
            junction_tuples = ast.literal_eval(tuples_row["junctions"].iloc[0])
        except Exception as e:
            raise ValueError(f'Error parsing junctions for gene: {tuples_row["geneName"]}')
        
        junctions = [
            data_model.RelativeJunction(
                start=int(s),
                end=int(e),
                weight=float(w)
            ) for s, e, w in junction_tuples
        ]

        return junctions
        
    # Parse gene annotation gtf
    logger.info("Reading gene records ...")
    genes_dict = read_genes(
        dataset_id=dataset_id,
        reference_id=reference_id,
        gene_annotation_file=gene_annotation_file,
    )
    logger.info(f"Found {len(genes_dict)} genes in the gene annotation file.")

    # Parse junction tuples from BED file
    logger.info("Parsing junction tuples ...")
    df = pd.read_csv(bed_file, sep="\t")

    junction_records = []
    for gene_id, gene_meta in genes_dict.items():
        junctions = parse_junctions(df, gene_id)

        if not junctions:  # Skips over genes not present in BED file (ie. whose junctions were filtered out during CZ pipeline)
            continue

        record = data_model.JunctionRecord(
            **gene_meta,
            celltype=celltypes,
            technology_name=technology_name,
            junctions=junctions,
        )
        junction_records.append(record)
    logger.info(f"Parsed junctions for {len(junction_records)} genes.")

    table: pa.Table = data_model.JunctionRecord.to_pyarrow_table(junction_records)
    pq.write_to_dataset(
        table,
        output_path,
        partition_cols=["is_long_read", "reference_id", "dataset_id"],
    )
    logger.info("Done!")

def get_parser(parser=None):
    import argparse
    import os
    from pathlib import Path

    def validate_input_file(filepath):
        """Validate that input file exists and is readable."""
        path = Path(filepath)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"File does not exist: {filepath}")
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"Path is not a file: {filepath}")
        if not os.access(filepath, os.R_OK):
            raise argparse.ArgumentTypeError(f"File is not readable: {filepath}")
        return str(path.resolve())

    def validate_output_path(filepath):
        """Validate that output path directory exists and is writable."""
        path = Path(filepath)
        parent_dir = path.parent
        
        if not parent_dir.exists():
            raise argparse.ArgumentTypeError(f"Output directory does not exist: {parent_dir}")
        if not os.access(parent_dir, os.W_OK):
            raise argparse.ArgumentTypeError(f"Output directory is not writable: {parent_dir}")
        return str(path.resolve())

    def validate_dataset_id(dataset_id):
        """Validate dataset ID format."""
        if not dataset_id:
            raise argparse.ArgumentTypeError("Dataset ID cannot be empty")
        if '/' in dataset_id:
            raise argparse.ArgumentTypeError("Dataset ID cannot contain '/' character")
        if len(dataset_id) > 100:
            raise argparse.ArgumentTypeError("Dataset ID cannot exceed 100 characters")
        return dataset_id

    def validate_reference_id(reference_id):
        """Validate reference ID format."""
        if not reference_id:
            raise argparse.ArgumentTypeError("Reference ID cannot be empty")
        if '/' in reference_id:
            raise argparse.ArgumentTypeError("Reference ID cannot contain '/' character")
        return reference_id
    
    description = """
Process gene and junction data to create junction records.

This script reads gene annotations and junction data to create
structured junction records saved as Parquet files. The output is partitioned by long-read flag,
reference ID, and dataset ID for efficient querying.

Required file formats:
- Gene annotation: GTF format with 'geneID' attribute
- Junction data: BED file outputted by CZ pipeline
"""

    if parser is None:
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawTextHelpFormatter,
        )
    else:
        parser = parser.add_parser(
            "commit-sr-records",
            description=description,
            formatter_class=argparse.RawTextHelpFormatter,
        )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    
    required.add_argument(
        '--dataset-id',
        type=validate_dataset_id,
        required=True,
        metavar='ID',
        help="""
        Unique identifier for this dataset. Used for partitioning output data.
        Cannot contain '/' character or exceed 100 characters.
        Examples: 'K562', 'patient_001', 'control_replicate_1'
        """
    )
    
    required.add_argument(
        '--output-path',
        type=validate_output_path,
        required=True,
        metavar='PATH',
        help="""
        Path where the output Parquet dataset will be written.
        Parent directory must exist and be writable.
        Output will be partitioned into subdirectories.
        Example: './data/isoforms.parquet'
        """
    )
    
    required.add_argument(
        '--gene-annotation',
        type=validate_input_file,
        required=True,
        metavar='GTF_FILE',
        help="""
        Path to gene annotation file in GTF format.
        Must contain 'geneID' attribute for each gene record.
        Genes should include 'exons' field for structural annotation.
        """
    )
    
    required.add_argument(
        '--bed-file',
        type=validate_input_file,
        required=True,
        metavar='BED_FILE',
        help="""
        Path to junctions tuples file in BED format.
        Must contain 'geneID' attribute.
        Must include 'junctions' field with (start, end, weight) tuples.
        """
    )

    # Optional arguments
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument(
        '--reference-id',
        type=validate_reference_id,
        default='unknown',
        metavar='ID',
        help="""
        Reference genome identifier for this data. Used for partitioning.
        Cannot contain '/' character. 
        Examples: 'hg38', 'grch38.gencode.v29', 'mm10'
        (default: %(default)s)
        """
    )

    optional.add_argument(
        '--is-single-cell',
        action='store_true',
        help="""
        Indicates that this is single-cell data.
        """
    )

    optional.add_argument(
        '--technology-name',
        type=str,
        help="""
        Indicates technology platform used to obtain RNA-sequencing data.
        """
    )

    def parse_celltypes(celltype_string):
        """Parse celltype string into dictionary of name:weight pairs."""
        if not celltype_string:
            return None
        
        try:
            celltypes = {}
            pairs = celltype_string.split(',')
            total_weight = 0.0
            
            for pair in pairs:
                if ':' not in pair:
                    raise argparse.ArgumentTypeError(f"Invalid celltype format: '{pair}'. Expected 'name:weight'")
                
                name, weight_str = pair.split(':', 1)
                name = name.strip()
                weight_str = weight_str.strip()
                
                if not name:
                    raise argparse.ArgumentTypeError("Celltype name cannot be empty")
                
                if '/' in name:
                    raise argparse.ArgumentTypeError(f"Celltype name '{name}' cannot contain '/' character")
                
                try:
                    weight = float(weight_str)
                except ValueError:
                    raise argparse.ArgumentTypeError(f"Invalid weight value: '{weight_str}'. Must be a number")
                
                if weight <= 0.0:
                    raise argparse.ArgumentTypeError(f"Weight for '{name}' must be positive, got {weight}")
                
                if weight > 1.0:
                    raise argparse.ArgumentTypeError(f"Weight for '{name}' must be ≤ 1.0, got {weight}")
                
                celltypes[name] = weight
                total_weight += weight
            
            if total_weight > 1.0:
                raise argparse.ArgumentTypeError(f"Total celltype weights cannot exceed 1.0, got {total_weight:.3f}")
            
            return celltypes
            
        except Exception as e:
            if isinstance(e, argparse.ArgumentTypeError):
                raise
            raise argparse.ArgumentTypeError(f"Error parsing celltypes: {e}")

    optional.add_argument(
        '--celltypes',
        type=parse_celltypes,
        default=None,
        metavar='CELLTYPE_PAIRS',
        help="""
        Cell type proportions for bulk data deconvolution as comma-separated name:weight pairs.
        Only used for bulk data (when --is-single-cell is NOT set).
        Weights must be positive numbers ≤ 1.0 and total must not exceed 1.0.
        
        Examples:
          --celltypes "T_cell:0.3,B_cell:0.2,NK_cell:0.1"
          --celltypes "CD4:0.4,CD8:0.3,Monocyte:0.2"
        
        If not provided for bulk data, deconvolution will be performed automatically.
        """
    )

    return parser

        
def main():
    import sys
    args = get_parser().parse_args()

    try:
        format_records(
            output_path=args.output_path,
            gene_annotation_file=args.gene_annotation,
            bed_file=args.bed_file,
            dataset_id=args.dataset_id,
            reference_id=args.reference_id,
            is_single_cell=args.is_single_cell,
            celltypes=args.celltypes,
            technology_name=args.technology_name
        )
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed with error: {e}")
        logger.error("Check input files and parameters.")
        sys.exit(1)


if __name__ == "__main__":
    main()