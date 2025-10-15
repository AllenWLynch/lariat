# AL - moved rules to junction_processing.smk
# Step 1 in CZ pipeline -- index BAM & regtools junctions extract
rule junctions_extract:
    input:
        bam=sample_bam
    output:
        DB_path_temp("intermediates/{dataset_id}/junctions.extract.bed")
    params:
        str_spec=get_sr_strand_specificity
    conda: "../envs/regtools.yaml"
    shadow: "shallow"
    shell:
        """
        samtools index {input.bam}
        regtools junctions extract -s {params.str_spec} {input.bam} > {output}
        """

# Step 3 in CZ pipeline -- regtools junctions annotate
rule junctions_annotate:
    input:
        junc_extr=rules.junctions_extract.output, # for series of rules, you can refer to the outputs of previous rules using rules.<rulename>.output! AL
        ref_genome=reference_genome,
        ref_annot=reference_annotation,
        fai=reference_fai,
    output:
        DB_path_temp("intermediates/{dataset_id}-{reference_id}/junctions.annotate.bed")
    conda: "../envs/regtools.yaml" # added conda env - need to go up one path to access envs directory
    shadow: "shallow" # the shadow directive makes it so that "side effect" files (like .bai, .fai, or anything else that is collateral output of a command)
                      # are written to a temp directory so they are cleaned up after the rule completes.
    log: "logs/junctions_annotate/{dataset_id}-{reference_id}.log"
    shell:
        """
        gzip -dc {input.ref_annot} > {input.ref_annot}.gff
        regtools junctions annotate {input.junc_extr} {input.ref_genome} {input.ref_annot}.gff > {output} 2> {log}
        """

# Step 4 of CZ pipeline -- extend TSS and TES of every gene by 512 bp on both sides & group all transcripts by gene
rule collapse_transcripts:
    input:
        ref_annot=reference_annotation,
        chromsizes=reference_chromsizes,
    output:
       DB_path_temp("intermediates/{reference_id}/annotated.collapsed.bed")
    conda: "../envs/gffread.yaml"
    params:
        scripts_dir=SCRIPTS_DIR
    shell:
        """
        gzip -dc {input.ref_annot} | 
            gffread --stream -W |
            bedtools slop -i - -g {input.chromsizes} -b 512 |
            python {params.scripts_dir}/collapse_transcripts.py > {output}
        """

# Step 5 of CZ pipeline -- intersect junction data w/ reference genome gene data
rule bedtools_intersect:
    input:
        junc_annot=rules.junctions_annotate.output,
        ref_annot_collapsed=rules.collapse_transcripts.output,
    output:
        DB_path_temp("intermediates/{dataset_id}-{reference_id}/junctions.intersect.bed")
    conda: "../envs/gffread.yaml" # has bedtools
    shell:
        """
        bedtools intersect -a {input.junc_annot} -b {input.ref_annot_collapsed} -wa -wb -s -f 1.0 | 
            cut -f1-14,19-21,24 > {output}
        """

def _get_gene_intersections(wildcards):
    return rules.bedtools_intersect.output[0].format(
        dataset_id=wildcards.dataset_id,
        reference_id=sample_ref_id(wildcards),
    )

# Step 6 of CZ pipeline -- calculate junction start & end offsets based on gene start index, store output in tuple form
rule calculate_offsets_tuples:
    input:
        junc_inter=_get_gene_intersections,
    output:
        bed_file=temp(partial_format(JUNCTIONS_PATH, bulk_level="bulk"))
    params:
        scripts_dir=SCRIPTS_DIR,
        known=config.junction_min_count_known,
        unknown=config.junction_min_count_unknown,
    shell:
        """
        python {params.scripts_dir}/calculate_junc_annot_offsets.py \
            --min-score-known {params.known} \
            --min-score-unknown {params.unknown} \
            --input {input.junc_inter} \
            --output {output.bed_file}
        """