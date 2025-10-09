import pydantic
from typing import Dict, Any, Optional, Union, Literal, Annotated
from pydantic import Field


class BamConfig(pydantic.BaseModel):
    bam_file: str = Field(
        ..., description="Path to the BAM file containing long-read alignments."
    )
    cage_file: Optional[str] = Field(None, description="Path to the CAGE TSV file.")
    umi_tag: Optional[str] = Field(
        None, description="Optional BAM tag for UMI sequences."
    )
    cell_barcode_tag: Optional[str] = Field(
        None, description="Optional BAM tag for cell barcodes."
    )
  
class DatasetBase(pydantic.BaseModel):
    link: Optional[str] = Field(
        None, description="Optional URL link to the dataset source."
    )
    description: Optional[str] = Field(
        None, description="Optional description of the dataset."
    )
    technology_name: str = Field(...)
    reference: str = Field(
        ..., description="Reference genome key, must not contain '/' character."
    )
    partition: Optional[str] = Field(
        None, description="Data partition (e.g., train, test, val) for the dataset."
    )

    @pydantic.field_validator("reference")
    @classmethod
    def validate_reference(cls, v):
        if "/" in v:
            raise ValueError("Reference key cannot contain '/' character")
        return v


class QuantitationConfig(pydantic.BaseModel):
    file: str = Field(...)
    weight_key: str = Field(...)
    transcript_id_key: str = Field(default="transcript_ID")
    celltype_key: str = Field(default="celltype")


class AnnotationConfig(pydantic.BaseModel):
    file: str = Field(...)
    transcript_id_key: str = Field(default="ID")
    gene_id_key: str = Field(default="geneID")


class LongReadPrecomputedConfig(pydantic.BaseModel):
    quantitation: QuantitationConfig = Field(
        ..., description="Transcript count TSV file."
    )
    annotation: AnnotationConfig = Field(
        ..., description="Transcript annotation GTF file."
    )
    bam_file: Optional[str] = Field(
        None,
        description="Optional path to the BAM file containing long-read alignments.",
    )


##
# Mixin classes
##
class BulkDataset(DatasetBase):
    celltypes: Optional[Union[str, Dict[str, float]]] = Field(
        None,
        description="Optional cell type proportions for deconvolution. If None, deconvolution will be performed.",
    )
    is_single_cell: Literal[False]

    @property
    def requires_deconvolution(self) -> bool:
        return self.celltypes is None

    @pydantic.field_validator("celltypes", check_fields=False)
    @classmethod
    def validate_bulk_celltypes(cls, v):
        if not isinstance(v, dict):
            return {v: 1.0} if v is not None else v

        for celltype, fraction in v.items():
            if not (0.0 < fraction <= 1.0):
                raise ValueError(
                    f"Cell type fraction for '{celltype}' must be between 0 and 1."
                )
            if "/" in celltype:
                raise ValueError(
                    f"Cell type name '{celltype}' cannot contain '/' character."
                )

        total_fraction = sum(v.values())
        if not (0.0 < total_fraction <= 1.0):
            raise ValueError(
                f"Total cell type fraction must be between 0 and 1, got {total_fraction}."
            )
        return v


class SingleCellDataset(DatasetBase):
    is_single_cell: Literal[True]

    @pydantic.field_validator("source_data", check_fields=False)
    @classmethod
    def validate_single_cell_source(cls, v):
        if isinstance(v, BamConfig):
            if v.umi_tag is None or v.cell_barcode_tag is None:
                raise ValueError(
                    "Single-cell datasets must specify both umi_tag and cell_barcode_tag in the source_data."
                )
            return v
        elif isinstance(v, LongReadPrecomputedConfig):
            if v.quantitation.celltype_key is None:
                raise ValueError(
                    "Single-cell precomputed datasets must specify celltype_key in the quantitation config."
                )
            return v
        else:
            raise ValueError(
                "source_data must be either BamConfig or LongReadPrecomputedConfig for single-cell datasets."
            )

    @property
    def requires_deconvolution(self) -> bool:
        return False


class LongReadDset(pydantic.BaseModel):
    source_data: Union[LongReadPrecomputedConfig, BamConfig] = Field(
        ..., description="Source data configuration for long-read datasets."
    )
    is_long_read: Literal[True]

    @property
    def is_precomputed(self) -> bool:
        return isinstance(self.source_data, LongReadPrecomputedConfig)


class ShortReadDataset(pydantic.BaseModel):
    source_data: BamConfig = Field(
        ..., description="Source data configuration for short-read datasets."
    )
    is_long_read: Literal[False]
    strand_specificity: str = Field(
        ..., description="Strand specificity of RNA library preparation (forward, reverse, unstranded)."
    )


class SingleCellLongReadDset(LongReadDset, SingleCellDataset):
    pass


class BulkLongReadDset(LongReadDset, BulkDataset):
    pass


class BulkShortReadDset(ShortReadDataset, BulkDataset):
    pass


class SingleCellShortReadDset(ShortReadDataset, SingleCellDataset):
    pass


ShortReadTypes = Annotated[
    Union[BulkShortReadDset, SingleCellShortReadDset],
    Field(discriminator="is_single_cell"),
]
LongReadTypes = Annotated[
    Union[BulkLongReadDset, SingleCellLongReadDset],
    Field(discriminator="is_single_cell"),
]

IsoformDset = Annotated[
    Union[LongReadTypes, ShortReadTypes], Field(discriminator="is_long_read")
]


class ReferenceConfig(pydantic.BaseModel):
    gene_annotation_file: str = Field(
        ..., description="Path to the gene annotation file (GTF format)."
    )
    fasta_file: str = Field(..., description="Path to the reference genome FASTA file.")
    species: str = Field(
        ...,
        description="<genus> <species> formatted species ID to use as key for species embeddings.",
    )
    transcript_id_key: str = Field("ID")
    gene_id_key: str = Field("geneID")

class IsoformDBCollection(pydantic.BaseModel):
    datasets: Dict[str, IsoformDset] = Field(...)
    references: Dict[str, ReferenceConfig] = Field(...)
    db_prefix: str = Field(
        "isoformDB/", description="Prefix path for the database files."
    )
    train_size: float = Field(
        0.8, description="Proportion of the dataset to be used for training."
    )
    celltype_embeddings: str = "TSF_embeddings/"

    @pydantic.field_validator("datasets")
    @classmethod
    def validate_dataset_keys(cls, v):
        for key in v.keys():
            if "/" in key:
                raise ValueError(f"Dataset key '{key}' cannot contain '/' character")
        return v

    @pydantic.field_validator("references")
    @classmethod
    def validate_reference_keys(cls, v):
        for key in v.keys():
            if "/" in key:
                raise ValueError(f"Reference key '{key}' cannot contain '/' character")
        return v

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        for sample_id, sample in self.datasets.items():
            if not sample.reference in self.references:
                raise ValueError(
                    f"Reference {sample.reference} for sample {sample_id} not found in dataset references."
                )
        
        # Validate short read specificity config values -CZ
        valid_values = {"forward", "reverse", "unstranded"}
        for sample_id, sample in self.datasets.items():
            if isinstance(sample, (BulkShortReadDset, SingleCellShortReadDset)):
                if sample.strand_specificity not in valid_values:
                    raise ValueError(
                        f"Invalid strand_specificity '{sample.strand_specificity}"
                        f"for sample '{sample_id}. Must be one of {valid_values}."
                    )

