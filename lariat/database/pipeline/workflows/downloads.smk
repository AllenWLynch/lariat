##
# Download rules
## 
DOWNLOAD="wget -O {output} {params.url}"

rule download_ref_annotation:
    output: DB_path("references/{reference_id}/gene_annotation.gtf.gz")
    params:
        url=lambda w : config.references[w.reference_id].gene_annotation_file
    shell:
        DOWNLOAD

rule download_ref_genome:
    output: DB_path("references/{reference_id}/genome.fa.gz")
    params:
        url=lambda w : config.references[w.reference_id].fasta_file
    shell:
        DOWNLOAD

rule download_annotation:
    output: DB_path("downloads/{dataset_id}/transcript_annotation.gtf.gz")
    params:
        url=lambda w : sample_config(w).source_data.annotation.file
    shell:
        DOWNLOAD

rule download_quantitation:
    output: DB_path("downloads/{dataset_id}/quantitation.tsv")
    params:
        url=lambda w : sample_config(w).source_data.quantitation.file
    shell:
        DOWNLOAD

rule download_bam:
    output: 
        bam=DB_path_temp("downloads/{dataset_id}/data.bam")
    params:
        url=lambda w : sample_config(w).source_data.bam_file
    shell:
        DOWNLOAD

rule download_cage:
    output: DB_path("downloads/{dataset_id}/data.cage.bed.gz")
    params:
        url=lambda w : sample_config(w).source_data.cage_file
    shell:
        DOWNLOAD