##
# SR BAM preprocessing
## 
rule process_bulk_sr_bam:
    input:
        bam=sample_bam,
        cage=sample_cage,
        genome=sample_genome,
        reference_annotation=sample_ref_annotation,
    output:
        junctions=temp(partial_format(JUNCTIONS_PATH, bulk_level="bulk"))


rule process_sc_sr_bam:
    input:
        bam=sample_bam,
        cage=sample_cage,
        genome=sample_genome,
        reference_annotation=sample_ref_annotation,
        celltype_clusters=celltype_clusters,
    params:
        umi_tag=lambda w: sample_config(w).source_data.umi_tag,
        cell_barcode_tag=lambda w: sample_config(w).source_data.cell_barcode_tag,
    output:
        junctions=temp(partial_format(JUNCTIONS_PATH, bulk_level="single-cell"))


rule aggregate_bulk_sr_counts:
    input:
        bam=sample_bam,
    output:
        gene_counts=temp(partial_format(BULK_COUNTS_PATH, read_type="short"))


rule generate_sc_sr_count_matrix:
    input:
        bam=sample_bam,
    output:
        count_matrix=temp(partial_format(COUNT_MATRIX_PATH, read_type="short"))
    params:
        umi_tag=lambda w: sample_config(w).source_data.umi_tag,
        cell_barcode_tag=lambda w: sample_config(w).source_data.cell_barcode_tag,



