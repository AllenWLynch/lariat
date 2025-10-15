
import os
from typing import Dict
import urllib
import json
from functools import partial, cached_property
import string
from typing import Dict
import pydantic
import encode

class PartialFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, f'{{{key}}}')  # keep placeholder if missing
        else:
            return string.Formatter.get_value(self, key, args, kwargs)

def partial_format(s, **kwargs):
    return PartialFormatter().format(s, **kwargs)


class EncodeDBConfig(pydantic.BaseModel):
    experiments: Dict[str, encode.EncodeExperimentConfig] = pydantic.Field(
        ..., description="Mapping from experiment accession to its configuration"
    )

SCRIPTS_DIR = config.pop("scripts")
config = EncodeDBConfig.model_validate(config)

def _first(x):
    return next(iter(x))

def experiment(wildcards):
    return config.experiments[wildcards.experiment]

def file(wildcards):
    return experiment(wildcards).files[wildcards.subfile]

def requires_aggregation(wildcards):
    return len(experiment(wildcards).files) > 1

ZARR_PATH = "tracks.zarr/{species}"
AGGREGATED_BIGWIG = "intermediates/aggregated/{experiment}.bigwig"
BIGWIG_FROM_BAM = "intermediates/from_bam/{experiment}.{subfile}.bigwig"
DOWNLOAD = "intermediates/downloaded/{experiment}.{subfile}.{extension}"

species = set(e.species for e in config.experiments.values())

wildcard_constraints:
    extension="(bigwig|bam)"

rule all:
    input:
        expand(ZARR_PATH, species=species)


def _get_chromsizes(wildcards):
    species = wildcards.get("species") or experiment(wildcards).species
    if species == "Homo_sapiens":
        return os.path.join(SCRIPTS_DIR, "hg38.chromsizes.txt")
    elif species == "Mus_musculus":
        return os.path.join(SCRIPTS_DIR, "mm10.chromsizes.txt")
    else:
        raise ValueError(f"Unknown species: {species}")

def _get_file_level_bigwig(experiment, file):
        return (
            partial_format(DOWNLOAD, extension="bigwig")
            if file.file_format=="bigWig" 
            else BIGWIG_FROM_BAM
        ).format(
            experiment=experiment.accession,
            subfile=file.accession
        )

def _get_experiment_level_bigwig(experiment):
    return (
        AGGREGATED_BIGWIG.format(experiment=experiment.accession)
        if len(experiment.files) > 1
        else _get_file_level_bigwig(
            experiment, 
            _first(experiment.files.values())
        )
    )

def _subset_species(wildcards):
    species = wildcards.species
    return list(filter(lambda e: e.species == species, config.experiments.values()))[:3]

def _get_params(wildcards):
    params = json.dumps([
        {
            "file": _get_experiment_level_bigwig(e),
            "accession": e.accession,
            "assay": e.assay_title,
            "target": e.target,
            "biosample_term_name": e.biosample_term_name,
            "biosample_classification": e.biosample_classification,
            "species": e.species,
        }
        for e in _subset_species(wildcards)
    ])
    with open(f"intermediates/{wildcards.species}.json", "w") as f:
        f.write(params)
    return f.name

def _get_input_bigwigs(wildcards):
    return list(map(_get_experiment_level_bigwig, _subset_species(wildcards)))

rule make_zarr:
    output:
        directory(ZARR_PATH)
    input:
        bigwigs=_get_input_bigwigs,
        chromsizes=_get_chromsizes
    log: "logs/make_zarr/{species}.log"
    conda: "envs/zarr.yaml"
    params:
        track_info=_get_params,
        scripts_dir=SCRIPTS_DIR
    threads: 1
    priority: 10 # highest priority to finish the zarr quickly
    shell:
        """
        python {params.scripts_dir}/write_zarr.py \
            {params.track_info} {output} \
            --chromsizes {input.chromsizes} \
            -@ {threads} \
            2> {log}
        """


def _get_subfiles(wildcards):
    e = experiment(wildcards)
    return [_get_file_level_bigwig(e, f) for f in e.files.values()]

rule aggregate_bigwigs:
    output:
        temp(AGGREGATED_BIGWIG)
    input:
        _get_subfiles,
    log: "logs/aggregate_bigwigs/{experiment}.log"
    conda: "envs/wiggletools.yaml"
    shadow: "shallow"
    priority:1 # high priority to clear the sub-bigwigs
    params:
        scripts_dir=SCRIPTS_DIR
    shell:
        """
        (
        python {params.scripts_dir}/chromsizes_from_bigwig.py {input} > {output}.chromsizes
        wiggletools sum {input} > {output}.wig
        wigToBigWig {output}.wig {output}.chromsizes {output}
        ) 2> {log}
        """


def _get_plus_shift(wildcards):
    e = experiment(wildcards)
    return 4 if e.assay_title == "ATAC-seq" else 0

def _get_minus_shift(wildcards):
    e = experiment(wildcards)
    return -4 if e.assay_title == "ATAC-seq" else 1

rule bigwig_from_bam:
    output:
        temp(BIGWIG_FROM_BAM)
    input:
        bam=partial_format(DOWNLOAD, extension="bam"),
        chromsizes=_get_chromsizes
    log: "logs/bigwig_from_bam/{experiment}.{subfile}.log"
    shadow: "shallow"
    conda: "envs/bigwig_from_bam.yaml"
    params:
        plus_shift=_get_plus_shift,
        minus_shift=_get_minus_shift,
    shell:
        """
        bedtools bamtobed -i {input.bam} | \
            awk -v OFS="\\t" '{{if ($6=="+"){{print $1,$2+{params.plus_shift},$3,$4,$5,$6}} else if ($6=="-") {{print $1,$2,$3+{params.minus_shift},$4,$5,$6}}}}' | \
            sort -k1,1 | \
            bedtools genomecov -bg -5 -i stdin -g {input.chromsizes} | \
            LC_COLLATE="C" sort -k1,1 -k2,2n \
            > {output}.tmp.bedgraph 2>> {log}

        bedGraphToBigWig {output}.tmp.bedgraph {input.chromsizes} {output} 2>> {log}
        """


def _get_url(wildcards):
    return file(wildcards).url

rule download:
    output:
        temp(DOWNLOAD)
    log: "logs/download/{experiment}.{subfile}.{extension}.log"
    params:
        url=_get_url
    shell:
        """
        wget -O {output} {params.url} 2> {log}
        """