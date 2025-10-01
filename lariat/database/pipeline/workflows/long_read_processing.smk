##
# LR BAM preprocessing
##
rule process_bulk_lr_bam:
    input:
        bam=sample_bam,
        cage=sample_cage,
        genome=sample_genome,
        reference_annotation=sample_ref_annotation,
    output:
        annotation=temp(partial_format(ANNOTATION_PATH, bulk_level="bulk")),
        quantitation=temp(partial_format(QUANTITATION_PATH, bulk_level="bulk")),


rule process_sc_lr_bam:
    input:
        bam=sample_bam,
        cage=sample_cage,
        genome=sample_genome,
        reference_annotation=sample_ref_annotation,
        celltype_clusters=celltype_clusters,
    output:
        annotation=temp(partial_format(ANNOTATION_PATH, bulk_level="single-cell")),
        quantitation=temp(partial_format(QUANTITATION_PATH, bulk_level="single-cell"))


rule aggregate_bulk_lr_counts:
    input:
        quantitation=sample_quantitation,
        annotation=sample_annotation,
    output:
        gene_counts=temp(partial_format(BULK_COUNTS_PATH, read_type="long"))


rule generate_sc_lr_count_matrix:
    input:
        bam=sample_bam,
    output:
        count_matrix=temp(partial_format(COUNT_MATRIX_PATH, read_type="long"))
    params:
        umi_tag=lambda w: sample_config(w).source_data.umi_tag,
        cell_barcode_tag=lambda w: sample_config(w).source_data.cell_barcode_tag,