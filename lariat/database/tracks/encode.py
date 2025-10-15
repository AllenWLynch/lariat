import datetime
from enum import IntEnum
from itertools import groupby, chain
from typing import Any, Callable, Dict, List, Tuple, Annotated
from functools import cache, partial
import requests
from collections import defaultdict
import pydantic

__all__ = ["EncodeExperimentConfig", "EncodeFileConfig", "get_encode_config"]

Experiment = Annotated[Dict[str, Any], "An experiment dictionary from ENCODE"]
File = Annotated[Dict[str, Any], "A file dictionary from ENCODE"]

cache_request = cache(requests.get)

not_accepted_terms = (
    "genetically modified",
    "arrested ",
    "treated "
)
allowed_assays = (
    "TF+ChIP-seq",
    "Histone+ChIP-seq",
    "total+RNA-seq",
    "polyA+plus+RNA-seq",
    "DNase-seq",
    "ATAC-seq",
    "eCLIP",
    "CAGE",
    "PRO-cap",
)
min_read_lengths = defaultdict(lambda : 50, {"DNase-seq": 36, "ATAC-seq": 45})
experiment_url = (
    "https://www.encodeproject.org/search/"
    "?type=Experiment"
    "&status=released"
    "&award.project=ENCODE"
    "&assembly=GRCh38"
    "&assembly=mm10"
    "&perturbed=false"
    "&field=replicates.library.biosample.donor.organism.scientific_name"
    "&field=accession"
    "&field=assay_title"
    "&field=target.label"
    "&field=biosample_ontology.term_name"
    "&field=biosample_ontology.classification"
    "&field=simple_biosample_summary"
    "&field=date_created"
    "&field=life_stage_age"
    "&field=audit"
    "&field=files.read_length"
    "&field=files.run_type"
    "&field=files.file_format"
    "&field=files.output_type"
    "&field=files.biological_replicates"
    "&field=files.technical_replicates"
    "&field=files.accession"
    "&field=files.preferred_default"
    "&field=files.quality_metrics.frip"
    "&field=files.href"
    "&field=files.date_created"
    "&field=files.derived_from"
    "&format=json"
    "&limit=1000000000"
) + "".join(f"&assay_title={assay}" for assay in allowed_assays)

class AuditFlag(IntEnum):
    FAIL = -1
    ERROR = 0
    NOT_COMPLIANT = 1
    WARNING = 2
    INTERNAL_ACTION = 3
    PASS = 4

audit_metadata: Dict[Tuple[str, str, str], bool] = {('INTERNAL_ACTION',
  'audit_analysis',
  'insufficient read depth for broad peaks control'): True,
 ('INTERNAL_ACTION', 'audit_analysis', 'multiple datasets'): True,
 ('INTERNAL_ACTION',
  'audit_antibody_characterization_status',
  'mismatched lane status'): True,
 ('INTERNAL_ACTION',
  'audit_antibody_characterization_unique_reviews',
  'duplicate lane review'): True,
 ('INTERNAL_ACTION',
  'audit_biosample',
  'duplicated genetic modifications'): True,
 ('INTERNAL_ACTION',
  'audit_biosample',
  'inconsistent BiosampleType term'): True,
 ('INTERNAL_ACTION',
  'audit_biosample',
  'mismatched genetic modifications'): True,
 ('INTERNAL_ACTION', 'audit_biosample', 'missing donor'): True,
 ('INTERNAL_ACTION', 'audit_biosample_type', 'NTR biosample'): True,
 ('INTERNAL_ACTION', 'audit_experiment', 'NTR biosample'): True,
 ('INTERNAL_ACTION',
  'audit_experiment',
  'biological replicates with identical biosample'): True,
 ('INTERNAL_ACTION',
  'audit_experiment',
  'experiment not submitted to GEO'): True,
 ('INTERNAL_ACTION', 'audit_experiment', 'inconsistent analysis files'): True,
 ('INTERNAL_ACTION', 'audit_experiment', 'inconsistent analysis status'): True,
 ('INTERNAL_ACTION', 'audit_experiment', 'inconsistent assay_term_name'): True,
 ('INTERNAL_ACTION',
  'audit_experiment',
  'inconsistent genetic modification targets'): True,
 ('INTERNAL_ACTION',
  'audit_experiment',
  'inconsistent genetic modifications'): True,
 ('INTERNAL_ACTION', 'audit_experiment', 'inconsistent internal tags'): True,
 ('INTERNAL_ACTION',
  'audit_experiment',
  'inconsistent mapped reads lengths'): True,
 ('INTERNAL_ACTION', 'audit_experiment', 'missing RIN'): True,
 ('INTERNAL_ACTION',
  'audit_experiment',
  'missing raw data in replicate'): True,
 ('INTERNAL_ACTION',
  'audit_experiment',
  'missing unfiltered alignments'): True,
 ('INTERNAL_ACTION', 'audit_experiment', 'mixed libraries'): True,
 ('INTERNAL_ACTION', 'audit_experiment', 'out of date analysis'): True,
 ('INTERNAL_ACTION',
  'audit_experiment',
  'sequencing runs labeled as technical replicates'): True,
 ('INTERNAL_ACTION', 'audit_experiment', 'unreplicated experiment'): True,
 ('INTERNAL_ACTION',
  'audit_experiment_released_with_unreleased_files',
  'mismatched file status'): True,
 ('INTERNAL_ACTION', 'audit_file', 'duplicate quality metric'): True,
 ('INTERNAL_ACTION', 'audit_file', 'inconsistent derived_from'): True,
 ('INTERNAL_ACTION', 'audit_file', 'missing derived_from'): True,
 ('INTERNAL_ACTION', 'audit_item_relations_status', 'mismatched status'): True,
 ('INTERNAL_ACTION', 'audit_item_status', 'mismatched status'): True,
 ('INTERNAL_ACTION', 'audit_status_replicate', 'mismatched status'): True,
 ('INTERNAL_ACTION',
  'audit_treatment_no_purpose',
  'missing treatment purpose'): True,
 ('WARNING', 'audit_analysis', 'borderline genes detected'): True,
 ('WARNING', 'audit_analysis', 'borderline mapping rate'): True,
 ('WARNING', 'audit_analysis', 'borderline microRNAs expressed'): True,
 ('WARNING', 'audit_analysis', 'borderline number of aligned reads'): True,
 ('WARNING', 'audit_analysis', 'borderline replicate concordance'): True,
 ('WARNING', 'audit_analysis', 'borderline sequencing depth'): True,
 ('WARNING', 'audit_analysis', 'high pct_unique_total_duplicates'): True,
 ('WARNING', 'audit_analysis', 'low coverage'): True,
 ('WARNING', 'audit_analysis', 'low intra/inter-chr PET ratio'): True,
 ('WARNING', 'audit_analysis', 'low lambda C conversion rate'): True,
 ('WARNING', 'audit_analysis', 'low pct_ligation_motif_present'): True,
 ('WARNING', 'audit_analysis', 'low pct_unique_hic_contacts'): True,
 ('WARNING',
  'audit_analysis',
  'low pct_unique_long_range_greater_than_20kb'): True,
 ('WARNING', 'audit_analysis', 'low read depth'): True,
 ('WARNING', 'audit_analysis', 'low replicate concordance'): True,
 ('WARNING', 'audit_analysis', 'low sequenced_read_pairs'): True,
 ('WARNING', 'audit_analysis', 'low spot score'): True,
 ('WARNING', 'audit_analysis', 'low total read pairs'): True,
 ('WARNING', 'audit_analysis', 'low total_unique reads'): True,
 ('WARNING', 'audit_analysis', 'mild to moderate bottlenecking'): True,
 ('WARNING', 'audit_analysis', 'missing footprints'): True,
 ('WARNING', 'audit_analysis', 'missing lambda C conversion rate'): True,
 ('WARNING', 'audit_analysis', 'moderate FRiP score'): True,
 ('WARNING', 'audit_analysis', 'moderate TSS enrichment'): True,
 ('WARNING', 'audit_analysis', 'moderate library complexity'): True,
 ('WARNING', 'audit_analysis', 'moderate number of reproducible peaks'): True,
 ('WARNING',
  'audit_experiment',
  'antibody characterized with exemption'): True,
 ('WARNING', 'audit_experiment', 'control low read depth'): True,
 ('WARNING',
  'audit_experiment',
  'improper control_type of control experiment'): True,
 ('WARNING', 'audit_experiment', 'inconsistent platforms'): True,
 ('WARNING', 'audit_experiment', 'lacking processed data'): True,
 ('WARNING', 'audit_experiment', 'low read length'): True,
 ('WARNING', 'audit_experiment', 'missing biosample characterization'): True,
 ('WARNING',
  'audit_experiment',
  'missing compliant biosample characterization'): True,
 ('WARNING',
  'audit_experiment',
  'missing control_type of control experiment'): True,
 ('WARNING',
  'audit_experiment',
  'missing genetic modification characterization'): True,
 ('WARNING', 'audit_experiment', 'missing spikeins'): True,
 ('WARNING', 'audit_experiment', 'mixed read lengths'): True,
 ('WARNING', 'audit_experiment', 'mixed run types'): True,
 ('WARNING',
  'audit_experiment',
  'unexpected target of control experiment'): True,
 ('WARNING', 'audit_file', 'inconsistent assembly'): True,
 ('WARNING', 'audit_file', 'inconsistent control read length'): True,
 ('WARNING', 'audit_file', 'inconsistent control run_type'): True,
 ('WARNING', 'audit_file', 'inconsistent paired_with'): True,
 ('WARNING', 'audit_file', 'matching md5 sums'): True,
 ('WARNING', 'audit_file', 'missing analysis_step_run'): True,
 ('WARNING', 'audit_file', 'missing controlled_by'): True,
 ('WARNING', 'audit_file', 'missing paired_with'): True,
 ('WARNING',
  'audit_fly_worm_donor_genotype_dbxrefs',
  'missing external identifiers'): True,
 ('WARNING',
  'audit_fly_worm_donor_genotype_dbxrefs',
  'missing genotype'): True,
 ('WARNING',
  'audit_genetic_modification_reagents',
  'missing genetic modification reagents'): True,
 ('NOT_COMPLIANT',
  'audit_analysis',
  'extremely low pct_ligation_motif_present'): False,
 ('NOT_COMPLIANT',
  'audit_analysis',
  'extremely low pct_unique_long_range_greater_than_20kb'): False,
 ('NOT_COMPLIANT', 'audit_analysis', 'insufficient coverage'): False,
 ('NOT_COMPLIANT',
  'audit_analysis',
  'insufficient number of aligned reads'): False,
 ('NOT_COMPLIANT',
  'audit_analysis',
  'insufficient number of reproducible peaks'): False,
 ('NOT_COMPLIANT', 'audit_analysis', 'insufficient read depth'): True,
 ('NOT_COMPLIANT',
  'audit_analysis',
  'insufficient replicate concordance'): False,
 ('NOT_COMPLIANT', 'audit_analysis', 'insufficient sequencing depth'): True,
 ('NOT_COMPLIANT', 'audit_analysis', 'low FRiP score'): True,
 ('NOT_COMPLIANT', 'audit_analysis', 'low TSS enrichment'): False,
 ('NOT_COMPLIANT', 'audit_analysis', 'low non-redundant PET'): False,
 ('NOT_COMPLIANT', 'audit_analysis', 'poor library complexity'): False,
 ('NOT_COMPLIANT', 'audit_analysis', 'severe bottlenecking'): False,
 ('NOT_COMPLIANT',
  'audit_experiment',
  'antibody not characterized to standard'): True,
 ('NOT_COMPLIANT',
  'audit_experiment',
  'control insufficient read depth'): True,
 ('NOT_COMPLIANT', 'audit_experiment', 'inconsistent age'): False,
 ('NOT_COMPLIANT', 'audit_experiment', 'insufficient read length'): True,
 ('NOT_COMPLIANT', 'audit_experiment', 'missing RNA fragment size'): True,
 ('NOT_COMPLIANT', 'audit_experiment', 'missing documents'): True,
 ('NOT_COMPLIANT', 'audit_experiment', 'missing input control'): True,
 ('NOT_COMPLIANT', 'audit_experiment', 'missing possible_controls'): True,
 ('NOT_COMPLIANT', 'audit_experiment', 'missing spikeins'): True,
 ('NOT_COMPLIANT',
  'audit_experiment',
  'partially characterized antibody'): True,
 ('NOT_COMPLIANT', 'audit_experiment', 'uncharacterized antibody'): True,
 ('NOT_COMPLIANT', 'audit_experiment', 'unreplicated experiment'): True,
 ('ERROR', 'audit_analysis', 'extremely low coverage'): False,
 ('ERROR', 'audit_analysis', 'extremely low read depth'): False,
 ('ERROR', 'audit_analysis', 'extremely low spot score'): False,
 ('ERROR', 'audit_analysis', 'missing footprints'): False,
 ('ERROR', 'audit_analysis', 'missing lambda C conversion rate'): False,
 ('ERROR', 'audit_biosample', 'missing donor'): False,
 ('ERROR', 'audit_biosample_type', 'inconsistent ontology term'): False,
 ('ERROR',
  'audit_dataset_with_uploading_files',
  'file validation error'): False,
 ('ERROR', 'audit_experiment', 'control extremely low read depth'): False,
 ('ERROR', 'audit_experiment', 'extremely low read length'): False,
 ('ERROR', 'audit_experiment', 'inconsistent donor'): False,
 ('ERROR', 'audit_experiment', 'inconsistent ontology term'): False,
 ('ERROR', 'audit_experiment', 'inconsistent target'): False,
 ('ERROR', 'audit_experiment', 'missing antibody'): False,
 ('ERROR',
  'audit_experiment',
  'missing compliant biosample characterization'): False,
 ('ERROR', 'audit_experiment', 'missing control alignments'): False,
 ('ERROR', 'audit_experiment', 'missing possible_controls'): False,
 ('ERROR', 'audit_experiment', 'missing queried_RNP_size_range'): False,
 ('ERROR',
  'audit_experiment',
  'not compliant biosample characterization'): False,
 ('ERROR', 'audit_file', 'inconsistent control'): False,
 ('ERROR', 'audit_file', 'inconsistent read count'): False,
 ('ERROR', 'audit_file', 'missing analysis_step_run'): False}

def get_audit_flag(audits: Dict[str, List[Dict[str, str]]]) -> Tuple[AuditFlag, str]:
    flags = []
    for level_name, level_audits in audits.items():
        for audit in level_audits:
            name = audit["name"]
            category = audit["category"]
            tolerated = audit_metadata.get((level_name, name, category), True)
            flags.append(
                (getattr(AuditFlag, level_name) if tolerated else AuditFlag.FAIL, category)
            )

    return min(flags, default=(AuditFlag.PASS, "No issues found"), key=lambda x: x[0])

def species_name(experiment: Experiment) -> str:
    return (
        experiment["replicates"][0]["library"]["biosample"]["donor"]["organism"]["scientific_name"]
        if experiment["replicates"] and experiment["replicates"][0]["library"]["biosample"].get("donor")
        else "unknown"
    )

def date_created(experiment: Experiment | File) -> datetime.datetime:
    return datetime.datetime.strptime(experiment["date_created"].split(":", 1)[0], "%Y-%m-%dT%H")

def frip(file: File) -> float:
    if not "quality_metrics" in file:
        return 0.0
    qm = file["quality_metrics"]
    for q in qm:
        if "frip" in q:
            return q["frip"]
    return 0.0

def groupkey(experiment: Experiment) -> Tuple:
    return (
        species_name(experiment),   
        experiment["biosample_ontology"]["term_name"], 
        experiment["biosample_ontology"]["classification"], 
        experiment["assay_title"], 
        experiment.get("target", {}).get("label", "None")
    )

def filter_experiments(experiment):
    files = experiment.get("files", [])
    read_lengths = {f.get("read_length", 0) for f in files}
    return (
        read_lengths
        and max(read_lengths) >= min_read_lengths[experiment["assay_title"]]
        and experiment["audit_flag"] > AuditFlag.FAIL
    )

def meets_filter_criteria(experiment):
    biosample_classification = experiment.get("biosample_ontology", {}).get("classification", "").lower()
    files = experiment.get("files", [])
    read_type = {f.get("run_type", 'single-ended') for f in files}

    return all([
        not any(term in experiment.get("simple_biosample_summary", "") for term in not_accepted_terms),
        experiment.get("audit_flag", AuditFlag.PASS) > AuditFlag.NOT_COMPLIANT,
        'paired-ended' in read_type,
        biosample_classification in {"primary cell", "tissue", "cell line", "whole organism"},
        date_created(experiment) > datetime.datetime(2020, 1, 1),
    ])

def experiment_priority(experiment: Experiment) -> Tuple:
    """
    Prioritize experiments according to the following criteria (in order of importance):
    1. Used non-genetically modified samples (this preference primarily impacts TF ChIP-seq selection). - already filtered out
    2. Had less severe audit flags (PASS > WARNING > NOT_COMPLIANT).
    3. Used paired-end sequencing.
    4. Derived from primary cells (vs. in vitro differentiated cells) or tissues (vs. organoids).
    5. Originated from more commonly profiled life stages (e.g., adult > embryonic > child).
    6. Were not from subcellular fractions.
    7. more recently created
    8. Prefer longer read lengths.
    """
    biosample_classification = experiment.get("biosample_ontology", {}).get("classification", "").lower()
    life_stage_age = experiment.get("life_stage_age", "").lower()
    files = experiment.get("files", [])
    read_lengths = {f.get("read_length", 0) for f in files}
    read_type = {f.get("run_type", 'single-ended') for f in files}

    return (
        not any(term in experiment.get("simple_biosample_summary", "") for term in not_accepted_terms),
        experiment.get("audit_flag", AuditFlag.PASS),                  # 2. Audit flag
        'paired-ended' in read_type,                                   # 3. Paired-end sequencing
        biosample_classification == "primary cell" or biosample_classification == "tissue",  # 4. Primary cells/tissues
        (
            4 if "adult" in life_stage_age else
            3 if "child" in life_stage_age else
            2 if "embryonic" in life_stage_age else
            1 if life_stage_age 
            else 0
        ),
        biosample_classification in {"whole organism", "cell line", "primary cell", "tissue"},  # 6. Subcellular fractions
        date_created(experiment),                                   # 7. More recently created
        max(read_lengths) if read_lengths else 0                      # Prefer longer read lengths
    )

@cache
def select_experiments() -> List[Experiment]:

    def _add_audit_info(experiment: Experiment) -> Experiment:
        audit_flag, description = get_audit_flag(experiment.get("audit", {}))
        return {
            **experiment,
            "audit_flag": audit_flag,
            "audit_description": description
        }
    
    experiments = cache_request(experiment_url).json()["@graph"]
    experiments = map(_add_audit_info, experiments)
    filtered_experiments = list(filter(filter_experiments, experiments))
    selected_experiments = []
    grouped_experiments = groupby(sorted(filtered_experiments, key=groupkey), key=groupkey)
    for _, group in grouped_experiments:
        group = list(group)
        ok_experiments = list(filter(meets_filter_criteria, group))
        if ok_experiments:
            selected_experiments.extend(ok_experiments)
        else:
            selected_experiments.append(max(group, key=experiment_priority))
        
    return selected_experiments

def _most_recent_files(files: List[File]) -> List[File]:
    def _key(f: File):
        return (f.get("file_format"), f.get("output_type"), f.get("biological_replicates"), f.get("technical_replicates"))
    return [max(fs, key=date_created) for _, fs in groupby(sorted(files, key=_key), _key)]

def _group_sorted(x: List[Any], key: Callable[[Any], Any]) -> List[List[Any]]:
    return [list(g) for _, g in groupby(sorted(x, key=key), key=key)]

def _select_rna_seq_files(files: List[File]) -> List[File]:
    files = [
        f for f in files 
        if (
            f.get("file_format") == "bigWig"
            and f.get("output_type") in {"plus strand signal of unique reads", "minus strand signal of unique reads"}
        )
    ]
    return _most_recent_files(files)

def _select_peak_files(frip_threshold: float, output_type: str, files: List[File]) -> List[File]:
    analysis_groups = [ # 1. group by replicate and get the most recent file of each type
        _most_recent_files(fs) 
        for fs in _group_sorted(
            files, 
            key=lambda f: (f["biological_replicates"], f["technical_replicates"])
        )
    ]

    analysis_groups = [ # 2. filter out groups that don't meet quality criteria
        ag for ag in analysis_groups
        if any(frip(f) >= frip_threshold for f in ag)
    ]

    files = [
        f for f in chain.from_iterable(analysis_groups)
        if f["output_type"] == output_type
    ]
    return files

assay_to_file_selector = {
    "total RNA-seq": _select_rna_seq_files,
    "polyA plus RNA-seq": _select_rna_seq_files,
    "CAGE": _select_rna_seq_files,
    "PRO-cap": _select_rna_seq_files,
    "eCLIP": _select_rna_seq_files,
    "DNase-seq": partial(_select_peak_files, 0.1, "alignments"),
    "ATAC-seq": partial(_select_peak_files, 0.1, "alignments"),
    "Histone ChIP-seq": partial(_select_peak_files, 0.01, "fold change over control"),
    "TF ChIP-seq": partial(_select_peak_files, 0.05, "fold change over control"),
}

def choose_files(experiment: Experiment) -> List[File]:
    return assay_to_file_selector[experiment["assay_title"]](experiment.get("files", []))


class EncodeFileConfig(pydantic.BaseModel):
    accession: str = pydantic.Field(..., description="Experiment accession")
    file_format: str = pydantic.Field(..., description="File format, e.g. bigWig")
    output_type: str = pydantic.Field(..., description="Output type, e.g. fold change over control")
    url: str = pydantic.Field(..., description="URL to download the file")
    biological_replicates: List[int] = pydantic.Field(..., description="List of biological replicate IDs")
    technical_replicates: List[int] = pydantic.Field(..., description="List of technical replicate IDs")


class EncodeExperimentConfig(pydantic.BaseModel):
    accession: str = pydantic.Field(..., description="Experiment accession")
    assay_title: str = pydantic.Field(..., description="Assay title")
    target: str = pydantic.Field(..., description="Target")
    species: str = pydantic.Field(..., description="Species")
    biosample_term_name: str = pydantic.Field(..., description="Biosample term name")
    biosample_classification: str = pydantic.Field(..., description="Biosample classification")
    audit_flag: int = pydantic.Field(4, description="Audit flag")
    audit_description: str = pydantic.Field(..., description="Audit description")
    files: Dict[str, EncodeFileConfig] = pydantic.Field(..., description="List of file URLs")

    # make sure the species names do not have spaces
    @pydantic.model_validator(mode="after")
    def check_species(self) -> "EncodeExperimentConfig":
        self.species = self.species.replace(" ", "_")
        return self

class NoFilesError(Exception):
    pass

def to_config(experiment: Experiment) -> EncodeExperimentConfig:
    files = choose_files(experiment)
    if not files:
        raise NoFilesError("No suitable files found")

    file_configs = {
        file["accession"] : EncodeFileConfig(
            accession=file["accession"],
            file_format=file["file_format"],
            output_type=file["output_type"],
            url="https://www.encodeproject.org" + file["href"],
            biological_replicates=file.get("biological_replicates", []),
            technical_replicates=file.get("technical_replicates", []),
        )
        for file in files
    }

    return EncodeExperimentConfig(
        accession=experiment["accession"],
        assay_title=experiment["assay_title"],
        target=experiment.get("target", {}).get("label", "None"),
        species=species_name(experiment),
        biosample_term_name=experiment["biosample_ontology"]["term_name"],
        biosample_classification=experiment["biosample_ontology"]["classification"],
        files=file_configs,
        audit_flag=experiment.get("audit_flag", AuditFlag.PASS),
        audit_description=experiment.get("audit_description", "No issues found"),
    )

def get_encode_config() -> Dict[str, EncodeExperimentConfig]:
    experiments = select_experiments()
    experiment_configs = {}
    for experiment in experiments:
        try:
            config = to_config(experiment)
            experiment_configs[experiment["accession"]] = config
        except NoFilesError:
            continue
    return experiment_configs

def main():
    import argparse
    import sys
    import json

    parser = argparse.ArgumentParser(description="Fetch and display ENCODE experiment configurations.")
    parser.add_argument(
        "--output","-o",
        type=argparse.FileType("w"), 
        default=sys.stdout,
        help="Output file to save the configuration (JSON format). If not provided, prints to stdout."
    )
    args = parser.parse_args()

    config = get_encode_config()
    json_config = {k: v.model_dump() for k, v in config.items()}
    with args.output as f:
        json.dump({"experiments": json_config}, f, indent=2)

if __name__ == "__main__":
    main()