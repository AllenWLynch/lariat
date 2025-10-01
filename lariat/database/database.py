from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Generator
import os
from functools import reduce, cached_property
from collections.abc import Iterable as _Iterable
import numpy as np
import json
from lariat.genome_utils import Region, Strand
from .data_model import IsoformRecord, JunctionRecord, DBRecord

if TYPE_CHECKING:
    import pyarrow.parquet as pq
    import pyarrow as pa

OptionalStrSequence = Optional[str | Iterable[str]]


class IsoformDB:
    """
    A little wrapper around the isoform database stored in Parquet format.
    Provides methods to load the database, iterate over records, and fetch reference sequences
    from the associated FASTA files.
    """

    def __init__(self, database_path: str, table: Optional["pa.Table"] = None) -> None:
        self.database_path = database_path
        self._table = table

    def write_vocab(self, vocab_file: str) -> None:
        with open(vocab_file, "w") as f:
            vocab_serialized = json.dumps(
                {
                    "technologies": list(self.technology_names),
                    "species": list(self.species_names),
                    "reference_id_to_species": self.reference_id_to_species,
                }
            )
            f.write(vocab_serialized)

    @property
    def table(self) -> "pa.Table":
        if self._table is None:
            self._table = self._load_table(self.database_path)
        return self._table

    @property
    def _table_path(self) -> str:
        return os.path.join(self.database_path, "database")

    def _load_table(self, database_path: str) -> "pa.Table":
        import pyarrow.parquet as pq

        return pq.read_table(self._table_path)

    def __repr__(self) -> str:
        return (
            f"IsoformDB(num_records={len(self)}, database_path='{self.database_path}')"
        )

    def select(
        self,
        gene_id: OptionalStrSequence = None,
        gene_name: OptionalStrSequence = None,
        reference_id: OptionalStrSequence = None,
        dataset_id: OptionalStrSequence = None,
        celltype: OptionalStrSequence = None,
        species: OptionalStrSequence = None,
        region: Optional[Region | Iterable[Region]] = None,
        min_celltype_fraction: Optional[float] = None,
        is_long_read: Optional[bool] = None,
        raise_if_empty: bool = True,
    ) -> IsoformDB:

        import pyarrow.compute as pc

        # Tiny wrappers to keep static analysis quiet (PyArrow compute has dynamic attrs)
        pc_equal = lambda a, b: pc.equal(a, b)  # type: ignore[attr-defined]
        pc_leq = lambda a, b: pc.less_equal(a, b)  # type: ignore[attr-defined]
        pc_geq = lambda a, b: pc.greater_equal(a, b)  # type: ignore[attr-defined]
        pc_and = lambda a, b: pc.and_(a, b)  # type: ignore[attr-defined]
        pc_or = lambda a, b: pc.or_(a, b)  # type: ignore[attr-defined]
        pc_is_in = lambda col, value_set: pc.is_in(col, value_set=value_set)  # type: ignore[attr-defined]

        def promote_to_iterable(val: Any | Iterable[Any]) -> Iterable[Any]:
            return (
                val
                if isinstance(val, _Iterable) and not isinstance(val, (str, bytes))
                else iter([val])
            )

        def combine_filters(colname: str, value: str | Iterable[str]):
            return reduce(
                pc_or,
                (pc_equal(self.table[colname], v) for v in promote_to_iterable(value)),
            )

        table = self.table
        # build an initial filter which everything satisfies
        filters = []
        if gene_id is not None:
            filters.append(combine_filters("gene_id", gene_id))
        if gene_name is not None:
            filters.append(combine_filters("gene_name", gene_name))
        if reference_id is not None:
            filters.append(combine_filters("reference_id", reference_id))
        if dataset_id is not None:
            filters.append(combine_filters("dataset_id", dataset_id))
        if is_long_read is not None:
            filters.append(pc_equal(table["is_long_read"], is_long_read))
        if species is not None:
            compatible_ref_ids = set()
            for sp in promote_to_iterable(species):
                if sp in self.species_to_reference_ids:
                    compatible_ref_ids.update(self.species_to_reference_ids[sp])
            if not compatible_ref_ids:
                if raise_if_empty:
                    raise ValueError(f"No reference IDs found for species {species}.")
                else:
                    return IsoformDB(database_path=self.database_path, table=table.slice(0, 0))
            filters.append(combine_filters("reference_id", compatible_ref_ids))            
        if region is not None:
            region_filter = lambda r: pc_and(
                pc_equal(table["chrom"], r.chrom),
                pc_and(pc_leq(table["start"], r.end), pc_geq(table["end"], r.start)),
            )
            filters.append(
                reduce(pc_or, (region_filter(r) for r in promote_to_iterable(region)))
            )
        if celltype is not None:
            celltype_filter = lambda ct: pc_is_in(table["celltypes"].field("keys"), ct)
            filters.append(
                reduce(
                    pc_or, (celltype_filter(ct) for ct in promote_to_iterable(celltype))
                )
            )
        if min_celltype_fraction is not None:
            filters.append(
                pc_geq(table["celltypes"].field("values"), min_celltype_fraction)
            )

        if not filters:
            raise ValueError("At least one filter must be provided.")

        combined_filter = reduce(pc_and, filters)
        new_table = table.filter(combined_filter)

        if new_table.num_rows == 0 and raise_if_empty:
            raise ValueError("No records match the specified filters.")

        return IsoformDB(database_path=self.database_path, table=new_table)

    def __len__(self) -> int:
        return self.table.num_rows

    def to_isoform_records(self) -> Generator[IsoformRecord | JunctionRecord]:
        return (
            DBRecord.from_pyarrow_record(row)
            for batch in self.table.to_batches()
            for row in batch.to_pylist()
        )

    def _get_ref_path(self, reference_id: str) -> str:
        ref_dir = os.path.join(self.database_path, "references", reference_id)
        if not os.path.exists(ref_dir):
            raise ValueError(
                f"Reference ID {reference_id} not found in database at {self.database_path}."
            )

        naked_path = os.path.join(ref_dir, "genome.fa")
        gz_path = naked_path + ".gz"
        if os.path.exists(naked_path):
            return naked_path
        elif os.path.exists(gz_path):
            return gz_path
        else:
            raise ValueError(
                f"No FASTA file found for reference ID {reference_id} in {ref_dir}."
            )

    def get_reference_sequence(
        self,
        record: IsoformRecord | Region,
        reference_id: Optional[str] = None,
        stranded: bool = True,
        tss_slop: int = 0,
        tes_slop: int = 0,
    ) -> str:
        from pyfaidx import Fasta

        region = record if isinstance(record, Region) else record.region
        region = region.slop_upstream(tss_slop).slop_downstream(tes_slop)

        reference_id = (
            reference_id if isinstance(record, Region) else record.reference_id
        )
        if reference_id is None:
            raise ValueError(
                "Reference ID must be provided either via argument or record."
            )

        with Fasta(self._get_ref_path(reference_id)) as fasta:

            segment = fasta[region.chrom][region.start : region.end]

            if stranded and region.strand is Strand.minus:
                segment = segment.reverse.complement  # type: ignore

            return str(segment.seq).upper()  # type: ignore

    def get_celltype_embeddings(
        self, celltype: str
    ) -> np.typing.NDArray[np.float32]: ...

    @cached_property
    def reference_id_to_species(self) -> Dict[str, str]:
        ref_dir = os.path.join(self.database_path, "references")
        ref_to_species = {}
        for reference_id in self.reference_ids:
            species_path = os.path.join(ref_dir, reference_id, "species.txt")
            if os.path.exists(species_path):
                with open(species_path, "r") as f:
                    species_name = f.read().strip()
                    ref_to_species[reference_id] = species_name
            else:
                ref_to_species[reference_id] = "unknown"
        return ref_to_species
    
    @cached_property
    def species_to_reference_ids(self) -> Dict[str, list[str]]:
        species_to_refs = defaultdict(list)
        for ref_id, species in self.reference_id_to_species.items():
            species_to_refs[species].append(ref_id)
        return species_to_refs

    @cached_property
    def technology_names(self) -> set[str]:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        table = pq.read_table(self._table_path, columns=["technology_name"])
        return set(pc.unique(table["technology_name"]).to_pylist())  # type: ignore

    @cached_property
    def species_names(self) -> set[str]:
        return set(self.reference_id_to_species.values())

    @cached_property
    def reference_ids(self) -> set[str]:
        ref_dir = os.path.join(self.database_path, "references")
        if not os.path.exists(ref_dir):
            raise ValueError(
                f"References directory not found in database at {self.database_path}."
            )
        return set(os.listdir(ref_dir))
