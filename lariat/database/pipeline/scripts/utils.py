import tempfile
from typing import Dict, Iterable, List, Tuple, Any, Optional
import subprocess
import os
import logging
from lariat.database import data_model
from lariat.genome_utils import GFFRecord
from lariat.database.pipeline.config_models import LongReadTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")


def parse_intervals(intervals: str, sep: str) -> List[Tuple[int, int]]:
    return [
        (int(start) - 1, int(end))
        for start, end in (c.split(sep) for c in intervals.split(","))
    ]


def _make_tlf(filename, output_file, cds_only=False):

    is_compressed = filename.endswith(".gz")
    base_command = "gffread -MK --stream --tlf " + ("-C" if cds_only else "")
    command = (
        "gzip -dc" if is_compressed else "cat"
    ) + f" {filename} | {base_command} > {output_file}"

    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(output_file):
            os.remove(output_file)
        raise Exception(
            f"\033[31mCommand '{e.cmd}' failed with return code {e.returncode}:\n{e.stderr}\033[0m"
        ) from e

    # check to see if anything was written
    if os.path.getsize(output_file) == 0:
        os.remove(output_file)
        raise Exception(
            f"\033[31mNo output was written to {output_file}. Please check that the input GFF file {filename} is valid.\033[0m"
        )


def read_as_tlf(filename: str, cds_only=False) -> Iterable[str]:

    with tempfile.NamedTemporaryFile(delete_on_close=True) as tlf_file:

        _make_tlf(filename, tlf_file.name, cds_only=cds_only)
        tlf_file.flush()

        for line in open(tlf_file.name, "r"):
            yield line.strip()


def read_genes(
    gene_annotation_file: str, # gtf/gff file
    dataset_id: Optional[str] = None,
    reference_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    # genes are already in TLF format
    genes: Iterable[GFFRecord] = map(
        GFFRecord.from_gff, open(gene_annotation_file, "r")
    )
    
    genes_dict: Dict[str, Dict[str, Any]] = {
        record.attributes["geneID"]: {
            "gene_id": record.attributes["geneID"],
            "gene_name": record.attributes.get("gene_name", None),
            "region": record.region,
            "reference_id": reference_id,
            "dataset_id": dataset_id,
            "annotated_exons": [
                data_model.Exon(*interval)
                for interval in parse_intervals(record.attributes["exons"], "-")
            ],
        }
        for record in genes
    }

    return genes_dict
