
checkpoint deconvolve_celltypes:
    input:
        gene_counts=BULK_COUNTS_PATH # just a tsv of gene name and counts
    params:
        tissue=lambda wildcards: config.datasets[wildcards.dataset].tissue, # the tissue name to query census with to get single-cell profiles
    output:
        celltype_fractions=temp(CELLTYPE_FRACTIONS_PATH)


rule fetch_embeddings:
    output:
        DB_path(config.celltype_embeddings, "predefined", "{celltype}.zarr")
    params:
        celltype=lambda wildcards: wildcards.celltype,

# deconvolve_celltypes > fetch_embeddings


rule get_cluster_TSF_embeddings:
    input:
        counts_matrix=CLUSTERED_COUNTS, # the count matrix has the clusters!
    output:
        directory(DB_path(config.celltype_embeddings, "consensus", "{dataset}"))

# cluster_cells > get_cluster_TSF_embeddings


def gather_celltype_embeddings(wildcards):
    embeddings = []
    for dataset_id, sample in config.datasets.items():

        if sample.is_single_cell: # single-cell datasets do not need deconvolution
            return []

        if sample.celltypes is not None: # user-specified
            celltypes = (
                sample.celltypes
                if not type(sample.celltypes) is str 
                else {sample.celltypes: 1.0}
            )
        else: # infer from deconvolution
            fractions_file = (
                checkpoints.deconvolve_celltypes
                .get(
                    dataset_id=dataset_id,
                    read_type="long" if sample.is_long_read else "short",
                )
                .output
                .celltype_fractions
            )
            with open(fractions_file) as cf:
                celltypes = json.load(cf)

        embeddings_path = lambda celltype: rules.fetch_embeddings.output[0].format(celltype=celltype)

        fetch_embeddings = [
            embeddings_path(celltype)
            for celltype in celltypes.keys()
        ]
        embeddings.extend(fetch_embeddings)

    return list(set(embeddings))


rule collect_celltype_embeddings:
    input:
        gather_celltype_embeddings
    output:
        touch(ZARR_COMMITTED)
