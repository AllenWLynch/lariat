import sys
import argparse
from collections import defaultdict
from typing import List, Dict, Generator
from lariat.genome_utils import GFFRecord
from lariat.database.pipeline.scripts.utils import read_as_tlf


def _rank_transcript(record: GFFRecord) -> int:
    return int(record.attributes["exonCount"])


def select_transcripts(
    tlf_file: str,
    gene_id_key: str = "geneID",
) -> Generator[GFFRecord, None, None]:
    transcripts_by_gene: Dict[str, List[GFFRecord]] = defaultdict(list)

    for line in read_as_tlf(tlf_file, cds_only=True):
        record = GFFRecord.from_gff(line)
        transcripts_by_gene[record.attributes[gene_id_key]].append(record)

    for transcripts in transcripts_by_gene.values():
        yield max(transcripts, key=_rank_transcript)


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Select canonical transcripts from a GFF file."
        )
    else:
        parser = parser.add_parser(
            "select-canonical-gene",
            description="Select canonical transcripts from a GFF file."
        )
    
    parser.add_argument("input", type=argparse.FileType("r"), help="Input GFF file")
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("w"),
        help="Output GFF file",
        default=sys.stdout,
    )
    parser.add_argument(
        "--gene-id-key",
        "-key",
        type=str,
        default="geneID",
        help="Key to use for gene IDs in the GFF attributes (default: 'geneID')",
    )
    return parser


def main():
    args = get_parser().parse_args()
    for transcript in select_transcripts(args.input.name, gene_id_key=args.gene_id_key):
        args.output.write(str(transcript) + "\n")


if __name__ == "__main__":
    main()
