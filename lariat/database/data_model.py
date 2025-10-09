from __future__ import annotations
from dataclasses import dataclass, asdict, field, replace
from typing import (
    List,
    Optional,
    Dict,
    Any,
    Tuple,
    Union,
    TYPE_CHECKING,
    NamedTuple,
    overload,
)
from functools import cache
from collections import defaultdict
from lariat.genome_utils import Region, Strand
from itertools import groupby
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pyarrow as pa
    from matplotlib.axes import Axes

__all__ = [
    "Uxid",
    "Exon",
    "Transcript",
    "RelativeTranscript",
    "IsoformRecord",
    "Junction",
    "RelativeJunction",
    "JunctionRecord",
]


class Uxid(NamedTuple):
    uxid: NDArray[np.uint8]
    start_pos: int
    weight: float

    def __str__(self) -> str:
        return "".join("uixd"[i] for i in self.uxid)


class Exon(NamedTuple):
    start: int
    end: int


@dataclass(unsafe_hash=True)
class Transcript:
    exons: List[Exon]  # absolute position
    weight: float = 1.0

    def to_junctions(self) -> List[Junction]:
        return [
            Junction(
                start=self.exons[i].end, end=self.exons[i + 1].start, weight=self.weight
            )
            for i in range(len(self.exons) - 1)
        ]

    @property
    def start(self) -> int:
        return self.exons[0].start

    @property
    def end(self) -> int:
        return self.exons[-1].end

    def to_intervals(self) -> List[Tuple[int, int]]:
        return [(e.start, e.end) for e in self.exons]

    def __post_init__(self):
        self.exons = sorted(self.exons, key=lambda e: e.start)

    def __repr__(self) -> str:
        return f"Transcript(weight={self.weight}, exons={self.to_intervals()})"

    def _repr_html_(self) -> str:
        exons_str = ", ".join(f"({e.start}, {e.end})" for e in self.exons)
        return f"""
        <div>
            <strong>Transcript</strong><br/>
            Weight: {self.weight}<br/>
            Exons: [{exons_str}]
        </div>
        """


@dataclass(unsafe_hash=True)
class RelativeTranscript(Transcript):
    """
    The same data elements as a transcript,
    but a new class name so that we know these are in relative coordinates.
    In addition, this class implements some other methods.
    """

    offset: int = 0

    def to_uxid(self) -> Uxid:
        if self.start < 0:
            raise ValueError(
                "Transcript starts with a negative coordinate, which is out-of-bounds for the given reference frame. You need to slop the IsoformRecord upstream."
            )

        rightmost = self.exons[-1].end
        uxid = np.zeros(rightmost, dtype=np.uint8)
        prev_exon = None
        for e in self.exons:
            uxid[e.start : e.end] = 2
            if prev_exon is not None:
                uxid[prev_exon.end : e.start] = 1
            prev_exon = e
        uxid[rightmost - 1] = 3
        return Uxid(uxid, self.offset, self.weight)

    @classmethod
    def from_uxid_string(
        cls, uxid: NDArray[np.uint8], start_pos: int = 0, weight: float = 1.0
    ) -> RelativeTranscript:
        exons = []
        consumed = 0
        for token, g in groupby(uxid, key=lambda x: min(x, 2)):
            length = sum(1 for _ in g)
            if token == 2:
                exons.append(Exon(start=consumed, end=consumed + length))
            consumed += length

        return cls(exons=exons, offset=start_pos, weight=weight)


class Junction(NamedTuple):
    start: int
    end: int
    weight: float = 1.0


class RelativeJunction(Junction):
    """
    This is the same as Junction, but a different class
    so we know these are in relative coordinates.
    """

    ...


@dataclass(unsafe_hash=True)
class DBRecord:
    region: Region
    annotated_exons: List[Exon]
    gene_id: Optional[str] = None  # ENTREZ
    reference_id: Optional[str] = None  # RefSeq or Ensembl
    dataset_id: Optional[str] = None  # e.g. SRA accession
    celltype: Dict[str, float] = field(
        default_factory=dict
    )  # celltype name -> proportion
    is_single_cell: bool = False
    gene_name: Optional[str] = None
    technology_name: Optional[str] = None

    def derive(self, **kwargs) -> DBRecord:
        return replace(self, **kwargs)

    @classmethod
    def from_pyarrow_record(
        cls, record: Dict[str, Any]
    ) -> IsoformRecord | JunctionRecord:
        base_kw = dict(
            gene_id=record["gene_id"],
            reference_id=record["reference_id"],
            dataset_id=record["dataset_id"],
            region=Region(
                chrom=record["chrom"],
                start=record["start"],
                end=record["end"],
                strand=Strand(record["strand"]),
            ),
            annotated_exons=[Exon(**e) for e in record["exons"]],
            celltype=dict(record["celltype"]),
            is_single_cell=record.get("is_single_cell", False),
            gene_name=record.get("gene_name", None),  # type: Optional[str]
            technology_name=record.get("technology_name", None),
        )

        if record["is_long_read"]:
            transcripts = [
                Transcript(weight=t["weight"], exons=[Exon(**e) for e in t["exons"]])
                for t in record["transcripts"]
            ]
            return IsoformRecord(transcripts=transcripts, **base_kw)
        else:
            junctions = [
                Junction(start=j["start"], end=j["end"], weight=j.get("weight", 1.0))
                for j in record["junctions"]
            ]
            return JunctionRecord(junctions=junctions, **base_kw)

    def as_pyarrow_record(self) -> Dict[str, Any]:
        for field in [
            "gene_id",
            "reference_id",
            "dataset_id",
            "region",
            "annotated_exons",
            "celltype",
        ]:
            if getattr(self, field) is None:
                raise ValueError(
                    f"Non-nullable field '{field}' cannot be None when converting to pyarrow record"
                )
        return {
            "gene_id": self.gene_id,
            "reference_id": self.reference_id,
            "dataset_id": self.dataset_id,
            "chrom": self.region.chrom,
            "start": self.region.start,
            "end": self.region.end,
            "strand": self.region.strand.value,
            "exons": [e._asdict() for e in self.annotated_exons],
            "transcripts": [
                {"weight": t.weight, "exons": [e._asdict() for e in t.exons]}
                for t in getattr(self, "transcripts", [])
            ],
            "junctions": [j._asdict() for j in getattr(self, "junctions", [])],
            "celltype": self.celltype,
            "gene_name": self.gene_name,
            "technology_name": self.technology_name,
            "is_long_read": getattr(self, "is_long_read", True),
            "is_single_cell": self.is_single_cell,
        }

    @classmethod
    @cache
    def PYARROW_SCHEMA(cls) -> "pa.Schema":
        import pyarrow as pa

        exon_struct = pa.struct(
            [pa.field("start", pa.int32()), pa.field("end", pa.int32())]
        )

        transcript_struct = pa.struct(
            [pa.field("weight", pa.float32()), pa.field("exons", pa.list_(exon_struct))]
        )

        junction_struct = pa.struct(
            [
                pa.field("start", pa.int32()),
                pa.field("end", pa.int32()),
                pa.field("weight", pa.float32()),
            ]
        )

        return pa.schema(
            [
                pa.field("gene_id", pa.string()),
                pa.field("reference_id", pa.string()),
                pa.field("dataset_id", pa.string()),
                # region fields
                pa.field("chrom", pa.string()),
                pa.field("start", pa.int32()),
                pa.field("end", pa.int32()),
                pa.field("strand", pa.int8()),
                # exon fields
                pa.field("exons", pa.list_(exon_struct)),
                # transcript fields
                pa.field("transcripts", pa.list_(transcript_struct)),
                pa.field("junctions", pa.list_(junction_struct)),
                # other metadata
                pa.field("celltype", pa.map_(pa.string(), pa.float16())),
                pa.field("is_long_read", pa.bool_()),
                pa.field("is_single_cell", pa.bool_()),
                pa.field("technology_name", pa.string(), nullable=True),
                pa.field("gene_name", pa.string(), nullable=True),
            ]
        )

    @classmethod
    def to_pyarrow_table(cls, genes: list[IsoformRecord] | list[JunctionRecord]) -> "pa.Table":
        import pyarrow as pa

        if not genes:
            raise ValueError("The genes list is empty")
        records = [g.as_pyarrow_record() for g in genes]
        return pa.Table.from_pylist(records, schema=cls.PYARROW_SCHEMA())

    @overload
    def _with_respect_to(self, intervals: List[Exon], region: Region) -> List[Exon]: ...

    @overload
    def _with_respect_to(
        self, intervals: List[Junction], region: Region
    ) -> List[Junction]: ...

    def _with_respect_to(
        self, intervals: Union[List[Exon], List[Junction]], region: Region
    ) -> Union[List[Exon], List[Junction]]:
        """Convert a list of exons and junctions to be with respect to the region."""

        def shift_interval(interval: Union[Exon, Junction]) -> Union[Exon, Junction]:
            new_region = self.region.derive(
                interval.start, interval.end
            ).with_respect_to(region)
            # Handle different interval types based on length
            if len(interval) >= 3:  # Junction has weight as 3rd element
                return type(interval)(new_region.start, new_region.end, *interval[2:])  # type: ignore
            else:  # Exon has only start and end
                return type(interval)(new_region.start, new_region.end)  # type: ignore

        return [shift_interval(interval) for interval in intervals]  # type: ignore


@dataclass(unsafe_hash=True)
class IsoformRecord(DBRecord):
    transcripts: List[Transcript] = field(default_factory=list)
    is_long_read = True

    def derive(self, **kwargs) -> IsoformRecord:
        return replace(self, **kwargs)

    @property
    def gene_count(self) -> float:
        return sum(t.weight for t in self.transcripts)

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, subscript: int | slice) -> IsoformRecord:
        transcripts_slice = self.transcripts[subscript]
        if isinstance(subscript, int):
            transcripts_slice = [transcripts_slice]
        return self.derive(transcripts=transcripts_slice)

    def to_relative_transcripts(self) -> List[RelativeTranscript]:
        """Convert all exons and transcripts to relative coordinates."""

        def _to_relative(transcript: Transcript) -> RelativeTranscript:
            new_exons: List[Exon] = self._with_respect_to(transcript.exons, self.region)  # type: ignore
            return RelativeTranscript(
                weight=transcript.weight, exons=new_exons, offset=0
            )

        return [_to_relative(t) for t in self.transcripts]

    def to_relative_exons(self) -> List[Exon]:
        result: List[Exon] = self._with_respect_to(self.annotated_exons, self.region)  # type: ignore
        return result

    def from_relative_transcripts(
        self, transcripts: List[RelativeTranscript]
    ) -> IsoformRecord:
        new_transcripts = [
            replace(t, exons=self._with_respect_to(t.exons, self.region))  # type: ignore
            for t in transcripts
        ]
        return replace(self, transcripts=new_transcripts)

    def to_junction_record(self) -> JunctionRecord:
        junctions_list = JunctionRecord.accumulate_junctions(
            [t.to_junctions() for t in self.transcripts]
        )
        return JunctionRecord(
            region=self.region,
            annotated_exons=self.annotated_exons,
            gene_id=self.gene_id,
            reference_id=self.reference_id,
            dataset_id=self.dataset_id,
            celltype=self.celltype,
            is_single_cell=self.is_single_cell,
            gene_name=self.gene_name,
            technology_name=self.technology_name,
            junctions=junctions_list,
        )

    def plot(self, **kwargs) -> "Axes":
        from lariat.plot.exon_plot import plot_exons

        return plot_exons(self, **kwargs)


@dataclass(unsafe_hash=True)
class JunctionRecord(DBRecord):
    junctions: List[Junction] = field(default_factory=list)
    is_long_read = False

    @staticmethod
    def accumulate_junctions(junctions_list: List[List[Junction]]) -> List[Junction]:
        junction_dict = defaultdict(float)
        for junctions in junctions_list:
            for j in junctions:
                junction_dict[(j.start, j.end)] += j.weight
        return [
            Junction(start=s, end=e, weight=w) for (s, e), w in junction_dict.items()
        ]

    def __post_init__(self):
        self.junctions = sorted(self.junctions, key=lambda j: (j.start, j.end))

    def to_relative_junctions(self) -> List[RelativeJunction]:
        new_junctions: List[Junction] = self._with_respect_to(self.junctions, self.region)  # type: ignore
        return [RelativeJunction(j.start, j.end, j.weight) for j in new_junctions]
