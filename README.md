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
