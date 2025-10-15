# Lariat

## Installation

First, download the repository from github:
```
$ git clone https://github.com/AllenWLynch/lariat.git
$ cd lariat
```

Next, start a new conda environment, and install the package in "pipeline" mode:
```
$ conda create --name lariat -c conda-forge -y python=3.12
$ conda activate lariat
$ pip install ".[pipeline]"
```

## For reference

The implementation of my database writer for long read data can be found at `lariat/database/pipeline/scripts/write_lr_records.py`.

The implementation of my database writer for short read data can be found at `lariat/database/pipeline/scripts/write_sr_records.py`.

## Using the DB

```
from lariat.database import IsoformDB

db = IsoformDB("isoformDB")

record = next(
    db.select(gene_name="MTOR", is_long_read=False) # for short-read data
    .to_isoform_records()
)
```

## Scaling up snakemake

Right now, the rules in our pipeline (for the most part) have the basic structure:

```python
rule example_rule:
    input: "path/to/{wildcard}/input"
    output: "path/to/{wildcard}/output"
    params:
        ...
    shell:
        """
        do-stuff {input} > {output}
        """
```

While this works fine locally, stderr outputs from each rule are spit into
the terminal, which makes debugging very difficult when running hundreds of 
jobs simulataneously. Next, we are running everything in the same python environment - 
which can get quite crowded. Finally, we don't have any resource specifications so
slurm won't schedule our jobs. Here's what the fully-realized scalable rule looks like:

```python
rule scalable_rule:
    input: "path/to/{wildcard}/input"
    output: "path/to/{wildcard}/output"
    params:
        ...
    conda: "envs/conda_env.yaml"
    log: "logs/scalable_rule/{wildcard}.log" # I always put the log file in "logs/{rule_name}/..."
    benchmark: "benchmarks/scalable_rule/{wildcard}.txt" # track resource usage for each rule
    shadow: "shallow" # execute the rule in a "shadow"-ed directory - side effects don't clog our file system
    threads: 1 # one thread
    resources:
        mem_mb=double_on_attempt(500), # when the rule fails, double the memory and time scheduled.
        runtime=double_on_attempt(300),
    shell:
        """
        do-stuff {input} > {output}
        """
```

A lot of this is boilerplate, but getting the resources set right can be challenging.
Finally, when you execute the pipeline in parallel mode, you can use:

```bash
$ lariat-compose --cluster --partition park "path/to/config.yaml"
```

which instructs snakemake to use the slurm executor. Make sure to install the executor first:
```bash
$ pip install snakemake-executor-plugin-slurm
```