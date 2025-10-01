from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps
from typing import Any, Iterable, Tuple

__all__ = [
    "Strand",
    "Region",
    "GFFRecord",
    "BED12Record",
]


class Strand(IntEnum):
    plus = 1
    minus = -1

    def __str__(self):
        return "+" if self == Strand.plus else "-"

    @classmethod
    def from_str(cls, strand_str: str) -> "Strand":
        if strand_str == "+":
            return Strand.plus
        elif strand_str == "-":
            return Strand.minus
        else:
            raise ValueError(f"Invalid strand: {strand_str}")


def _relative_op(func):
    @wraps(func)
    def wrapper(self: "Region", *args, **kwargs):
        return func(self.to_abs_coords(), *args, **kwargs).to_abs_coords()

    return wrapper


@dataclass
class Region:
    chrom: str
    start: int
    end: int
    strand: Strand = Strand.plus

    def __post_init__(self):
        self.start = int(self.start)
        self.end = int(self.end)

    @staticmethod
    def _reflect(region: "Region") -> "Region":
        return Region(
            region.chrom, -region.end, -region.start, Strand(-1 * region.strand)
        )

    def __len__(self):
        return abs(self.end - self.start)

    def to_abs_coords(self):
        return (
            self._reflect(self)
            if (self.strand == Strand.minus) ^ (self.end < 0)
            else self
        )

    def __sub__(self, x: int):
        return Region(self.chrom, self.start - x, self.end - x, self.strand)

    def __add__(self, x: int):
        return Region(self.chrom, self.start + x, self.end + x, self.strand)

    @_relative_op
    def __lshift__(self, x: int):
        return self - x

    @_relative_op
    def __rshift__(self, x: int):
        return self + x

    @_relative_op
    def slop_upstream(self, x: int):
        return Region(self.chrom, self.start - x, self.end, self.strand)

    @_relative_op
    def slop_downstream(self, x: int):
        return Region(self.chrom, self.start, self.end + x, self.strand)

    def slop(self, x: int) -> "Region":
        """
        Apply slop to both upstream and downstream.
        """
        return self.derive(
            start=max(self.start - x, 0),
            end=self.end + x,
        )

    def overlaps(self, other: "Region") -> bool:
        """
        Check if this region overlaps with another region.
        """
        if self.chrom != other.chrom:
            return False
        return not (self.end < other.start or self.start > other.end)

    def merge(self, other: "Region", stranded=False) -> "Region":

        if not self.overlaps(other):
            raise ValueError("Regions do not overlap, cannot merge.")
        if stranded and self.strand != other.strand:
            raise ValueError(
                "Stranded regions with different strands cannot be merged."
            )

        return self.derive(min(self.start, other.start), max(self.end, other.end))

    def __or__(self, other: "Region") -> "Region":
        """
        Merge two regions if they overlap.
        """
        return self.merge(other, stranded=True)

    def __and__(self, other: "Region") -> "Region":
        """
        Find the intersection of two regions.
        """
        if not self.overlaps(other):
            raise ValueError("Regions do not overlap, cannot find intersection.")

        return self.derive(
            start=max(self.start, other.start),
            end=min(self.end, other.end),
        )

    def __eq__(self, other) -> bool:
        """
        Check if two regions are equal.
        """
        if not isinstance(other, Region):
            return False
        return (
            self.chrom == other.chrom
            and self.start == other.start
            and self.end == other.end
            and self.strand == other.strand
        )

    def __gt__(self, other: "Region") -> bool:
        """
        Check if this region is greater than another region.
        Comparison is based on chrom, start, end, and strand.
        """
        if not isinstance(other, Region):
            return NotImplemented
        return self.chrom > other.chrom or (
            self.chrom == other.chrom and self.start > other.start
        )

    def derive(self, start: int, end: int) -> "Region":
        """
        Derive a new region from the current one with the specified start and end.
        """
        return Region(chrom=self.chrom, start=start, end=end, strand=self.strand)

    def with_respect_to(self, other: "Region") -> "Region":
        transform = (
            (lambda x: x) if other.strand == Strand.plus else (lambda x: x._reflect(x))
        )
        other = other.to_abs_coords()
        out = transform(self)

        return out.derive(start=out.start - other.start, end=out.end - other.start)

    @classmethod
    def coalesce(cls, regions: list["Region"], stranded=False) -> list["Region"]:
        """
        Merge a list of regions into a single region.
        """
        if not regions:
            raise ValueError("Cannot merge an empty list of regions.")

        regions = sorted(regions, key=lambda r: (r.chrom, r.start, r.end))
        merged_regions = [regions[0]]
        for current in regions[1:]:
            last = merged_regions[-1]
            if last.overlaps(current):
                merged_regions[-1] = last.merge(current)
            else:
                merged_regions.append(current)

        return sorted(merged_regions, key=lambda r: (r.chrom, r.start, r.end))


@dataclass(unsafe_hash=True)
class GFFRecord:
    region: Region
    source: str
    type: str
    score: float
    phase: str
    attributes: dict[str, Any] = field(default_factory=dict)

    GFF_COLS = [
        "chromosome",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    ]

    def __post_init__(self):
        self.score = float(self.score) if self.score != "." else 0.0

        for key, value in self.attributes.items():
            if value == ".":
                self.attributes[key] = None
                continue

            try:
                self.attributes[key] = int(value)
                continue
            except ValueError:
                pass

            try:
                self.attributes[key] = float(value)
                continue
            except ValueError:
                pass

    def __str__(self):
        return "\t".join(
            map(
                str,
                [
                    self.region.chrom,
                    self.source,
                    self.type,
                    self.region.start + 1,  # GFF is 1-based
                    self.region.end,
                    self.score if self.score != 0.0 else ".",
                    self.region.strand,
                    self.phase,
                    ";".join(f"{str(k)}={str(v)}" for k, v in self.attributes.items()),
                ],
            )
        )

    @classmethod
    def from_gff(cls, line: str) -> "GFFRecord":

        record: dict[str, str] = dict(zip(GFFRecord.GFF_COLS, line.strip().split("\t")))

        attributes = record.pop("attributes")
        attributes = dict([x.split("=", 1) for x in attributes.strip(";").split(";")])
        start = int(record.pop("start")) - 1
        end = int(record.pop("end"))
        strand = Strand.from_str(record.pop("strand"))

        region = Region(
            chrom=record.pop("chromosome"), start=start, end=end, strand=strand
        )

        score = record.pop("score")
        score = float(score) if score != "." else 0.0

        return cls(
            region=region,
            source=record["source"],
            type=record["type"],
            score=score,
            phase=record["phase"],
            attributes=attributes,
        )


@dataclass
class BED12Record:
    chrom: str
    start: int
    end: int
    name: str = "."
    score: float = 0.0
    strand: Strand = Strand.plus
    thick_start: int = 0
    thick_end: int = 0
    item_rgb: str = "0,0,0"
    block_count: int = 1
    block_sizes: list[int] = field(default_factory=lambda: [0])
    block_starts: list[int] = field(default_factory=lambda: [0])

    @classmethod
    def from_bed12(cls, line: str) -> "BED12Record":
        fields = line.strip().split("\t")
        if len(fields) < 12:
            raise ValueError("BED12 record must have at least 12 fields.")

        chrom = fields[0]
        start = int(fields[1])
        end = int(fields[2])
        name = fields[3] if fields[3] != "." else "."
        score = float(fields[4]) if fields[4] != "." else 0.0
        strand = Strand(fields[5]) if fields[5] in ["+", "-"] else Strand.plus
        thick_start = int(fields[6])
        thick_end = int(fields[7])
        item_rgb = fields[8]
        block_count = int(fields[9])
        block_sizes = list(map(int, fields[10].split(",")))
        block_starts = list(map(int, fields[11].split(",")))

        return cls(
            chrom=chrom,
            start=start,
            end=end,
            name=name,
            score=score,
            strand=strand,
            thick_start=thick_start,
            thick_end=thick_end,
            item_rgb=item_rgb,
            block_count=block_count,
            block_sizes=block_sizes,
            block_starts=block_starts,
        )

    def segments(self) -> Iterable[Tuple[str, int, int]]:
        for start, size in zip(self.block_starts, self.block_sizes):
            yield self.chrom, self.start + start, self.start + start + size

    def __len__(self) -> int:
        return sum(self.block_sizes)

    def __str__(self) -> str:
        return "\t".join(
            map(
                str,
                [
                    self.chrom,
                    self.start,
                    self.end,
                    self.name,
                    self.score,
                    self.strand,
                    self.thick_start,
                    self.thick_end,
                    self.item_rgb,
                    self.block_count,
                    ",".join(map(str, self.block_sizes)),
                    ",".join(map(str, self.block_starts)),
                ],
            )
        )

    @classmethod
    def from_regions(
        cls, regions: list[Region], name: str = ".", score: float = 0.0
    ) -> "BED12Record":
        if not regions:
            raise ValueError("Cannot create BED12Record from an empty list of regions.")

        chrom = regions[0].chrom
        start = min(region.start for region in regions)
        end = max(region.end for region in regions)

        block_sizes = [len(region) for region in regions]
        block_starts = [region.start - start for region in regions]

        strand = regions[0].strand

        return cls(
            chrom=chrom,
            start=start,
            end=end,
            name=name,
            score=score,
            strand=strand,
            block_sizes=block_sizes,
            block_starts=block_starts,
        )

    def as_gff_records(self) -> Iterable[GFFRecord]:
        """
        Convert the BED12 record to GFFRecord(s).
        Each segment in the BED12 record is converted to a GFFRecord.
        """
        yield GFFRecord(
            region=Region(
                chrom=self.chrom, start=self.start, end=self.end, strand=self.strand
            ),
            source="BED12",
            type="gene",
            score=self.score,
            phase=".",
            attributes={
                "ID": self.name,
            },
        )

        for i, (chrom, start, end) in enumerate(self.segments()):
            yield GFFRecord(
                region=Region(chrom=chrom, start=start, end=end, strand=self.strand),
                source="BED12",
                type="exon",
                score=self.score,
                phase=".",
                attributes={
                    "Parent": self.name,
                    "ID": f"{self.name}.exon{i+1}",
                },
            )
