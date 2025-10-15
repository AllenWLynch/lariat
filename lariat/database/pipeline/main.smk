import os
from typing import Dict
import urllib
import json
from functools import partial
import string
import config_models
from lariat.genome_utils import StrandSpecificity

SCRIPTS_DIR = config.pop("scripts")
config = config_models.IsoformDBCollection.model_validate(config)

wildcard_constraints:
    bulk_level="bulk|single-cell",
    read_type="long|short",
    dataset_id="[^./]+",
    reference_id=r"[^./]+(?:\.[^./]+)*",
    celltype="[^/]+",

class PartialFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, f'{{{key}}}')  # keep placeholder if missing
        else:
            return string.Formatter.get_value(self, key, args, kwargs)

def partial_format(s, **kwargs):
    return PartialFormatter().format(s, **kwargs)

def DB_path(*filenames):
    return os.path.join(config.db_prefix, *filenames)

def DB_path_temp(*filenames):
    return temp(DB_path(*filenames))

def double_on_attempt(base_resource):
    def resource_function(wildcards, attempt):
        return base_resource * (2 ** (attempt - 1))
    return resource_function
    
##
# Define the interfaces
##
JUNCTIONS_PATH = DB_path("processed/{dataset_id}/short.{bulk_level}.junctions.bed") # short read only
ANNOTATION_PATH = DB_path("processed/{dataset_id}/long.{bulk_level}.transcript_annotation.gtf.gz") # long read only
QUANTITATION_PATH = DB_path("processed/{dataset_id}/long.{bulk_level}.quantitation.tsv") # long read only
COUNT_MATRIX_PATH = DB_path("processed/{dataset_id}/{read_type}.single-cell.count_matrix.h5ad") # single-cell only
BULK_COUNTS_PATH = DB_path("processed/{dataset_id}/{read_type}.bulk.gene_counts.tsv") # bulk only
ZARR_COMMITTED = DB_path(".celltypes-committed")
CELLTYPE_FRACTIONS_PATH = DB_path("processed/{dataset_id}/{read_type}.bulk.celltype_fractions.json")
CELLTYPE_CLUSTERS_PATH = DB_path("processed/{dataset_id}/{read_type}.single-cell.clusters.tsv") # single-cell only
DATASET_COMMITTED = DB_path("database/is_long_read={is_long_read}/reference_id={reference_id}/dataset_id={dataset_id}/")
REF_COMMITTED = DB_path("references/{reference_id}/species.txt")

rule all:
    input:
        (
            [
                DATASET_COMMITTED.format(
                    is_long_read=str(sample.is_long_read).lower(),
                    reference_id=sample.reference,
                    dataset_id=dataset_id
                )
                for dataset_id, sample in config.datasets.items()
            ]
            + [
                REF_COMMITTED.format(
                    reference_id=reference_id
                )
                for reference_id in config.references.keys()
            ]
        )


def is_url(filename: str) -> bool:
    return urllib.parse.urlparse(filename).scheme in ["http", "https", "ftp"]

def file_or_download(filename, alt):
    return alt if is_url(filename) else filename

def symlink_or_download(filename, alt):
    if is_url(filename):
        return alt
    # else make a symlink

    # ensure the alt has the same compression status as the filename
    is_gzipped = filename.endswith(".gz")
    if not is_gzipped and alt.endswith(".gz"):
        alt = alt.removesuffix(".gz")

    dirname = os.path.dirname(alt)
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
        
    if os.path.exists(alt):
        os.remove(alt)
    os.symlink(os.path.abspath(filename), alt)
    return alt

def sample_config(wildcards):
    return config.datasets[wildcards.dataset_id]

def reference_config(wildcards):
    return config.references[sample_config(wildcards).reference]

def sample_ref_id(wildcards):
    return sample_config(wildcards).reference

def get_sr_strand_specificity(wildcards):
    str_spec = sample_config(wildcards).strand_specificity
    if str_spec == "forward":
        return str(StrandSpecificity.forward)
    elif str_spec == "reverse":
        return str(StrandSpecificity.reverse)
    else:
        return str(StrandSpecificity.unstranded)

def get_bulk_level(wildcards):
    return "single-cell" if sample_config(wildcards).is_single_cell else "bulk"

def get_read_type(wildcards):
    return "long" if sample_config(wildcards).is_long_read else "short"

include: "workflows/downloads.smk"

##
# Fetch functions
## 
def reference_annotation(wildcards):
    return symlink_or_download(
        config.references[wildcards.reference_id].gene_annotation_file,
        rules.download_ref_annotation.output[0].format(reference_id=wildcards.reference_id)
    )

def reference_genome(wildcards):
    return symlink_or_download(
        config.references[wildcards.reference_id].fasta_file,
        rules.download_ref_genome.output[0].format(reference_id=wildcards.reference_id)
    )

def reference_annotation(wildcards): # For rules that don't pass dataset_id as a wildcard -CZ
    return symlink_or_download(
        config.references[wildcards.reference_id].gene_annotation_file,
        rules.download_ref_annotation.output[0].format(reference_id=wildcards.reference_id)
    )

def sample_genome(wildcards):
    ref_id = sample_config(wildcards).reference
    return symlink_or_download(
        config.references[ref_id].fasta_file,
        rules.download_ref_genome.output[0].format(reference_id=ref_id)
    )

def sample_ref_annotation(wildcards):
    ref_id = sample_config(wildcards).reference
    return symlink_or_download(
        config.references[ref_id].gene_annotation_file,
        rules.download_ref_annotation.output[0].format(reference_id=ref_id)
    )

def sample_bam(wildcards):
    return file_or_download(
        sample_config(wildcards).source_data.bam_file,
        rules.download_bam.output
    )

def sample_cage(wildcards):
    cage = sample_config(wildcards).source_data.cage_file
    return file_or_download(cage, rules.download_cage.output)

def sample_quantitation(wildcards):
    sample = sample_config(wildcards)
    if not sample.is_precomputed:
        return QUANTITATION_PATH.format(
            dataset_id=wildcards.dataset_id,
            bulk_level=get_bulk_level(wildcards),
        )
    quantitation = sample.source_data.quantitation.file
    return file_or_download(quantitation, rules.download_quantitation.output)

def sample_annotation(wildcards):
    sample = sample_config(wildcards)
    if isinstance(sample, config_models.LongReadDset):
        if not sample.is_precomputed:
            return ANNOTATION_PATH.format(
                dataset_id=wildcards.dataset_id,
                bulk_level=get_bulk_level(wildcards),
            )
    annotation = sample.source_data.annotation.file
    return file_or_download(annotation, rules.download_annotation.output)

def sample_junctions(wildcards):
    return JUNCTIONS_PATH.format(
        dataset_id=wildcards.dataset_id,
        reference_id=sample_ref_id(wildcards),
        bulk_level=get_bulk_level(wildcards),
    )

def print_return(fn):
    def wrapper(wildcards):
        try:
            result = fn(wildcards)
        except Exception as e:
            print(f"Error in {fn.__name__} with wildcards={wildcards}: {repr(e)}")
            raise
        print(f"{fn.__name__} -> {result}; wildcards={wildcards}")
        return result
    return wrapper

# AL - moved genome indexing to its own rule, we now require 
# indexing to commit the reference.
rule index_genome:
    input:
        fasta=reference_genome
    output:
        fai=DB_path("references/{reference_id}/genome.fa.fai"),
        chromsizes=DB_path("references/{reference_id}/chromsizes.txt")
    conda: "envs/samtools.yaml"
    shell:
        """
        samtools faidx {input.fasta} -o {output.fai}
        cut -f1,2 {output.fai} > {output.chromsizes}
        """

def reference_fai(wildcards):
    return rules.index_genome.output.fai.format(reference_id=wildcards.reference_id)

def reference_chromsizes(wildcards):
    return rules.index_genome.output.chromsizes.format(reference_id=wildcards.reference_id)

def reference_data(wildcards):
    return {
        "fasta": reference_genome(wildcards),
        "fai": reference_fai(wildcards),
        "annotation": reference_annotation(wildcards),
        "chromsizes": reference_chromsizes(wildcards),
    }

rule commit_reference:
    input: 
        lambda wildcards: reference_data(wildcards).values()
    output: REF_COMMITTED
    params:
        species_name=lambda wildcards: config.references[wildcards.reference_id].species,
    run:
        with open(output[0], 'w') as f:
            f.write(params.species_name + '\n')


rule cluster_cells:
    input:
        count_matrix=COUNT_MATRIX_PATH,
    output:
        cluster_ids=DB_path_temp(CELLTYPE_CLUSTERS_PATH)

def celltype_clusters(wildcards):
    return CELLTYPE_CLUSTERS_PATH.format(
        dataset_id=wildcards.dataset_id,
        read_type=get_read_type(wildcards),
    )

include: "workflows/long_read_processing.smk"
include: "workflows/junction_processing.smk"
# include: "workflows/short_read_processing.smk"
# include: "workflows/deconvolution.smk"

##
# Ingestion
## 
rule make_intervals:
    input: reference_annotation
    output: DB_path("references/{reference_id}/cannonical_genes.gff")
    params:
        scripts_dir=SCRIPTS_DIR
    resources:
        mem_mb=2000,
        time="00:30:00",
        cpus_per_task=2
    shell:
        """
        python {params.scripts_dir}/select_canonical_gene.py {input} > {output}
        """

def get_intervals(wildcards):
    return rules.make_intervals.output[0].format(reference_id=sample_ref_id(wildcards))

def celltype_fractions(wildcards):
    sample = sample_config(wildcards)
    if not sample.requires_deconvolution:
        return []
    return CELLTYPE_FRACTIONS_PATH.format(
        dataset_id=wildcards.dataset_id,
        read_type=get_read_type(wildcards),
    )

def format_celltypes(celltypes: Dict[str, float]) -> str:
        return "--celltypes " + ",".join([f"{ct}:{weight}" for ct, weight in celltypes.items()]) + " \\\n"

def format_lr_argstring(wildcards):
    sample = sample_config(wildcards)
    reference_id = sample_ref_id(wildcards)

    source_data = sample.source_data
    quant_config = (
        source_data.quantitation
        if isinstance(source_data, config_models.LongReadPrecomputedConfig)
        else config_models.QuantitationConfig()
    )

    argstring = f"""--reference-id {reference_id} \\
            --dataset-id {wildcards.dataset_id} \\
            --weight-key {quant_config.weight_key} \\
            --transcript-id-key {quant_config.transcript_id_key} \\
            """

    if sample.is_single_cell:
        argstring += f"""--is-single-cell
            --celltype-key {quant_config.celltype_key} \\
            """
    elif sample.requires_deconvolution:
        fractions_file = CELLTYPE_FRACTIONS_PATH.format(
            dataset_id=wildcards.dataset_id,
            read_type=get_read_type(wildcards),
        )

        with open(fractions_file) as cf:
            celltypes = json.load(cf)
        
        argstring += format_celltypes(celltypes)
    else:
        argstring += format_celltypes(sample.celltypes)
        
    return argstring

rule write_lr_records:
    input:
        isoforms=sample_annotation,
        quantitation=sample_quantitation,
        intervals=get_intervals,
        celltypes=celltype_fractions,
    output:
        directory(partial_format(DATASET_COMMITTED, is_long_read="true")),
    params:
        database_path=DB_path("database/"),
        scripts_dir=SCRIPTS_DIR,
        argstring=format_lr_argstring,
    priority: 1000
    shell:
        """
        python {params.scripts_dir}/write_lr_records.py \\
            --gene-annotation {input.intervals} \\
            --quantitation {input.quantitation} \\
            --transcript-annotation {input.isoforms} \\
            --output-path {params.database_path} \\
            {params.argstring}
        """

def format_sr_argstring(wildcards):
    sample = sample_config(wildcards)
    reference_id = sample_ref_id(wildcards)

    argstring = f"""--reference-id {reference_id} \\
            --dataset-id {wildcards.dataset_id} \\
            """
    
    if sample.is_single_cell:
        argstring += "--is-single-cell" 
    
    argstring += format_celltypes(sample.celltypes)
    argstring += f"--technology-name {sample.technology_name}"
        
    return argstring

# Read junction tuples BED file and write into DB as JunctionRecords -CZ
rule write_sr_records:
    input:
        intervals=get_intervals,
        celltypes=celltype_fractions,
        bed_file=sample_junctions
    output:
        directory(partial_format(DATASET_COMMITTED, is_long_read="false"))
    params:
        database_path=DB_path("database/"),
        scripts_dir=SCRIPTS_DIR,
        argstring=format_sr_argstring,
    priority: 1000
    shell:
        """
        python {params.scripts_dir}/write_sr_records.py \\
            --gene-annotation {input.intervals} \\
            --output-path {params.database_path} \\
            --bed-file {input.bed_file} \\
            {params.argstring}
        """
