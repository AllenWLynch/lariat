import glob
import os
from typing import Iterator
import numpy as np
from functools import partial
from itertools import cycle
import io
import msgpack
from tarfile import TarInfo, TarFile
from tqdm import tqdm
import logging
from lariat.database.data_model import DBRecord
from lariat.database import IsoformDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")

__all__ = ["append_shard"]


def _write_webdataset_record(
    record: DBRecord,
    *,
    id_generator: Iterator[int],
    tar_writer: TarFile,
):
    def _append_to_tar(data, path):
        info = TarInfo(name=path)
        info.size = len(data)
        tar_writer.addfile(info, io.BytesIO(data))

    assert record.gene_id is not None
    prefix = (
        f"{next(id_generator)}/{record.dataset_id}_{record.gene_id.replace(".", "-")}"
    )
    data = record.as_pyarrow_record()
    transcripts_data = msgpack.packb(data, use_bin_type=True)
    assert transcripts_data is not None, "Failed to serialize transcripts with msgpack"
    _append_to_tar(transcripts_data, f"{prefix}.transcripts.msg")


def _write_shard(
    db: IsoformDB,
    shard_name: str,
    seed: int = 42,
) -> None:

    random_state = np.random.RandomState(seed)
    id_generator = iter(
        cycle(random_state.permutation(10_000))
    )  # Randomized ID generator
    write_record = partial(
        _write_webdataset_record,
        id_generator=id_generator,
    )

    with TarFile.open(shard_name, "w") as tar_writer:
        for record in tqdm(
            db.to_isoform_records(),
            desc="Writing records",
            unit="record",
            total=len(db),
        ):
            write_record(record, tar_writer=tar_writer)


def append_shard(shard_prefix: str, db_subset: IsoformDB, split="train") -> None:

    dirname, _ = os.path.split(shard_prefix)
    os.makedirs(dirname, exist_ok=True)

    # pad the shard number to 6 digits, leading zeros
    c = len(glob.glob(f"{shard_prefix}{split}-*.tar"))
    shard_name = f"{shard_prefix}{split}-{c:06d}.tar"
    logger.info(f"Writing shard: {shard_name} with {len(db_subset)} records")

    try:
        _write_shard(db_subset, shard_name, seed=c)  # Different seed for each shard
    except Exception as e:
        if os.path.exists(shard_name):
            os.remove(shard_name)  # Remove incomplete shard
        raise e
