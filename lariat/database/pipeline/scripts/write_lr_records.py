from typing import Any, Dict, List, Iterable, Tuple, Optional
import csv
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from dataclasses import replace

from lariat.database import data_model
from lariat.genome_utils import GFFRecord
from lariat.database.pipeline.config_models import QuantitationConfig
from lariat.database.pipeline.scripts.utils import (
    read_as_tlf,
    parse_intervals,
    logger,
    read_genes
)

def read_transcripts(
    transcript_annotation_file: str,
    genes_dict: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, data_model.Transcript], Dict[str, str]]:

    transcripts_dict: Dict[str, data_model.Transcript] = {}
    transcripts_to_gene: Dict[str, str] = {}

    transcript_records: Iterable[GFFRecord] = map(
        GFFRecord.from_gff, read_as_tlf(transcript_annotation_file)
    )
    gene_id_misses = 0
    for record in transcript_records:

        transcript_id = record.attributes["ID"]
        gene_id = record.attributes["geneID"]

        if not gene_id in genes_dict:
            gene_id_misses += 1
            continue

        transcript = data_model.Transcript(
            exons=[
                data_model.Exon(*interval)
                for interval in parse_intervals(record.attributes["exons"], "-")
            ]
        )
        transcripts_dict[transcript_id] = transcript
        transcripts_to_gene[transcript_id] = gene_id

    if gene_id_misses > 0:
        logger.info(
            f"{gene_id_misses} transcripts were skipped because their gene IDs were not in the gene annotation file."
        )

    if len(transcripts_dict) == 0:
        raise ValueError("No transcripts found in the transcript annotation file.")

    logger.info(
        f"Found {len(transcripts_dict)} transcripts in the transcript annotation file."
    )

    return (transcripts_dict, transcripts_to_gene)


def read_quantitation(
    quantitation_config: QuantitationConfig,
    genes_dict: Dict[str, Dict[str, Any]],
    transcripts_to_gene: Dict[str, str],
    is_single_cell: bool = False,
) -> Dict[str, Dict[Optional[str], Dict[str, float]]]:

    quant_dict: Dict[str, Dict[Optional[str], Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    transcript_id_misses = 0

    for row in csv.DictReader(open(quantitation_config.file, "r"), delimiter="\t"):

        transcript_id = row[quantitation_config.transcript_id_key]
        weight = float(row[quantitation_config.weight_key])
        celltype_id = (
            row[quantitation_config.celltype_key] if is_single_cell else None
        )

        if transcript_id not in transcripts_to_gene:
            transcript_id_misses += 1
            continue

        gene_id = transcripts_to_gene[transcript_id]
        quant_dict[gene_id][celltype_id][transcript_id] += weight

    if transcript_id_misses > 0:
        logger.info(
            f"{transcript_id_misses} quantitation records were skipped because their transcript IDs were not found in the transcript annotation file."
        )

    fraction_quantified = len(quant_dict) / len(genes_dict)
    logger.info(
        f"Found quantification data for {100*fraction_quantified:.2f}% of genes ({len(quant_dict)}/{len(genes_dict)})"
    )

    if fraction_quantified < 0.25:
        logger.warning(
            "Less than 25% of genes have quantification data. Please check that the gene IDs in the gene annotation file match those in the transcript annotation file."
        )

    return quant_dict


def format_records(
    *,
    output_path: str,
    gene_annotation_file: str,
    transcript_annotation_file: str,
    quantitation_file: str,
    weight_key: str,
    dataset_id: Optional[str] = None,
    reference_id: Optional[str] = None,
    transcript_id_key: str = "transcript_ID",
    celltype_key: str = "celltype",
    is_single_cell: bool = False,
    celltypes: Optional[Dict[str, float]] = None,
):

    quantitation_config = QuantitationConfig(
        file=quantitation_file,
        weight_key=weight_key,
        transcript_id_key=transcript_id_key,
        celltype_key=celltype_key,
    )

    logger.info("Reading gene records ...")
    genes_dict = read_genes(
        dataset_id=dataset_id,
        reference_id=reference_id,
        gene_annotation_file=gene_annotation_file,
    )
    logger.info(f"Found {len(genes_dict)} genes in the gene annotation file.")

    logger.info("Reading transcript records ...")
    transcripts_dict, transcripts_to_gene = read_transcripts(
        transcript_annotation_file=transcript_annotation_file,
        genes_dict=genes_dict,
    )

    logger.info("Reading quantitation data ...")
    quant_dict = read_quantitation(
        quantitation_config=quantitation_config,
        is_single_cell=is_single_cell,
        genes_dict=genes_dict,
        transcripts_to_gene=transcripts_to_gene,
    )

    logger.info("Constructing IsoformRecords and writing to Parquet ...")
    isoform_records: List[data_model.IsoformRecord] = [
        data_model.IsoformRecord(
            **gene_meta,
            celltype=(
                {celltype_id: 1.0}
                if is_single_cell and celltype_id is not None
                else celltypes or {}
            ),
            transcripts=[
                replace(transcripts_dict[transcript_id], weight=weight)
                for transcript_id, weight in transcript_weights.items()
            ],
        )
        for gene_id, gene_meta in genes_dict.items()
        for celltype_id, transcript_weights in quant_dict[gene_id].items()
    ]

    table: pa.Table = data_model.IsoformRecord.to_pyarrow_table(isoform_records)
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
Process gene and transcript annotations with quantitation data to create isoform records.

This script reads gene annotations, transcript annotations, and quantitation data to create
structured isoform records saved as Parquet files. The output is partitioned by long-read flag,
reference ID, and dataset ID for efficient querying.

Required file formats:
- Gene annotation: GTF format with 'geneID' attribute
- Transcript annotation: GTF format with 'ID' and 'geneID' attributes, plus 'exons' field
- Quantitation: TSV format with transcript IDs and quantitation values
"""

    if parser is None:
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawTextHelpFormatter,
        )
    else:
        parser = parser.add_parser(
            "commit-lr-records",
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
        '--transcript-annotation',
        type=validate_input_file,
        required=True,
        metavar='GTF_FILE',
        help="""
        Path to transcript annotation file in GTF format.
        Must contain 'ID' (transcript ID) and 'geneID' attributes.
        Must include 'exons' field with exon coordinates.
        """
    )
    
    required.add_argument(
        '--quantitation',
        type=validate_input_file,
        required=True,
        metavar='TSV_FILE',
        help="""
        Path to quantitation file in tab-separated format.
        Must contain columns for transcript IDs and quantitation values.
        For single-cell data, should include cell type/barcode information.
        """
    )
    
    required.add_argument(
        '--weight-key',
        type=str,
        required=True,
        metavar='COLUMN',
        help="""
        Column name in quantitation file containing the weight/count values.
        Common values: 'TPM', 'FPKM', 'count', 'abundance', 'est_counts'
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
        '--transcript-id-key',
        type=str,
        default='transcript_ID',
        metavar='COLUMN',
        help="""
        Column name in quantitation file containing transcript identifiers.
        Must match the 'ID' attribute in transcript annotation file.
        (default: %(default)s)
        """
    )
    
    optional.add_argument(
        '--celltype-key',
        type=str,
        default='celltype',
        metavar='COLUMN',
        help="""
        Column name in quantitation file containing cell type information.
        Only used when --is-single-cell flag is set.
        (default: %(default)s)
        """
    )
    
    optional.add_argument(
        '--is-single-cell',
        action='store_true',
        help="""
        Indicates that this is single-cell data with cell-type specific quantitation.
        When set, the script will use the --celltype-key column to group data by cell types.
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
            dataset_id=args.dataset_id,
            output_path=args.output_path,
            gene_annotation_file=args.gene_annotation,
            transcript_annotation_file=args.transcript_annotation,
            quantitation_file=args.quantitation,
            weight_key=args.weight_key,
            reference_id=args.reference_id,
            transcript_id_key=args.transcript_id_key,
            celltype_key=args.celltype_key,
            is_single_cell=args.is_single_cell,
            celltypes=args.celltypes,
        )
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed with error: {e}")
        logger.error("Check input files and parameters.")
        sys.exit(1)


if __name__ == "__main__":
    main()
