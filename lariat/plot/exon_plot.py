from typing import Dict, List, Optional, Tuple, Iterable, Union, Literal
from functools import partial
from itertools import groupby
from matplotlib.pylab import record
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy import argsort
from matplotlib import colors as mcolors
from lariat.database.data_model import (
    IsoformRecord,
    RelativeTranscript,
    Exon,
    JunctionRecord,
    RelativeJunction,
    Junction,
)


def _plot_junctions(
    ax: Axes,
    junctions: List[RelativeJunction | Junction],
    color: str = "C1",
    max_arc_height: float = 3.0,
    min_arc_height: float = 0.5,
    linewidth: float = 1,
    alpha: float = 0.75,
    height_offset: float = 0.0,
    **kwargs,
) -> Axes:
    """
    An Arc-style plot which visualizes splicing junctions on a given Axes.
    The arcs connect the donor and acceptor sites of each junction.
    The height of each arc is proportional to the junction's weight.
    """

    if not junctions:
        return ax

    relative_junctions = junctions
    # Get the range of weights to normalize arc height
    weights = [junction.weight for junction in relative_junctions]
    min_weight = min(weights)
    max_weight = max(weights)
    weight_range = max_weight - min_weight if max_weight > min_weight else 1.0

    base_y = 0  # Position arcs above the current content

    for junction in relative_junctions:
        # Normalize weight for scaling arc height
        normalized_weight = (
            (junction.weight - min_weight) / weight_range if weight_range > 0 else 0.5
        )

        # Scale arc height based on junction weight
        arc_height = min_arc_height + normalized_weight * (
            max_arc_height - min_arc_height
        )

        # Calculate arc parameters
        center_x = (junction.start + junction.end) / 2
        arc_width = junction.end - junction.start

        # Create a semi-circular arc using Arc patch
        # The arc goes from start to end with the specified height
        arc = patches.Arc(
            xy=(center_x, base_y + height_offset),  # Center of the arc
            width=arc_width,
            height=arc_height * 2,  # Height is diameter, so multiply by 2
            angle=0,
            theta1=0,  # Start angle (bottom left)
            theta2=180,  # End angle (bottom right) - creates upper semicircle
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            **kwargs,
        )

        ax.add_patch(arc)

    return ax


def _add_scalebar(
    ax: Axes, gene_body_length: int, y_position: float = 0.05, color: str = "black"
) -> None:
    """Add a non-intrusive scalebar indicating gene body size in kilobases."""
    # Calculate appropriate scalebar length (round to nice numbers)
    scalebar_length = max((gene_body_length // 10) // 5000 * 5000, 500)

    # Position scalebar in bottom right
    # xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate positions (10% margin from right edge, at specified y_position from bottom)
    bar_end_x = gene_body_length
    bar_start_x = bar_end_x - scalebar_length
    bar_y = ylim[1] + 1

    # Draw the scalebar
    ax.plot(
        [bar_start_x, bar_end_x],
        [bar_y, bar_y],
        color=color,
        linewidth=2,
        solid_capstyle="butt",
    )

    # Add text label
    bar_center_x = (bar_start_x + bar_end_x) / 2
    label = (
        f"{scalebar_length // 1000} kb"
        if scalebar_length >= 1000
        else f"{scalebar_length} bp"
    )
    ax.text(
        bar_center_x,
        bar_y + 0.25,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        color=color,
    )


def _get_transcripts_order(transcripts: Iterable[RelativeTranscript]) -> List[int]:
    import numpy as np
    from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering

    # Materialize iterable to allow multiple passes
    tx_list = list(transcripts)
    n = len(tx_list)
    if n == 0:
        return []

    # Precompute sorted intervals and total exon lengths per transcript
    intervals: List[List[Tuple[int, int]]] = []
    lengths: List[int] = []
    for t in tx_list:
        ivals = t.to_intervals()
        ivals.sort(key=lambda x: x[0])
        intervals.append(ivals)
        lengths.append(sum(e - s for s, e in ivals))

    def intersection_len(ai: List[Tuple[int, int]], bi: List[Tuple[int, int]]) -> int:
        i = j = 0
        inter = 0
        while i < len(ai) and j < len(bi):
            a_s, a_e = ai[i]
            b_s, b_e = bi[j]
            s = a_s if a_s > b_s else b_s
            e = a_e if a_e < b_e else b_e
            if e > s:
                inter += e - s
            # Advance the one that ends first
            if a_e < b_e:
                i += 1
            else:
                j += 1
        return inter

    # Build pairwise Jaccard similarity matrix efficiently
    pairwise = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        pairwise[i, i] = 1.0
        ai = intervals[i]
        li = lengths[i]
        for j in range(i + 1, n):
            bj = intervals[j]
            lj = lengths[j]
            inter = intersection_len(ai, bj)
            union = li + lj - inter
            jac = (inter / union) if union > 0 else 0.0
            pairwise[i, j] = jac
            pairwise[j, i] = jac

    d = 1.0 - pairwise
    # Convert square distance matrix to condensed vector for scipy.cluster.hierarchy
    from scipy.spatial.distance import squareform

    y = squareform(d, checks=False)
    Z = linkage(y, method="average")
    ordering = leaves_list(optimal_leaf_ordering(Z, y))
    return ordering


def _plot_transcript(
    *,
    exons: List[Exon],
    y: float,
    height: float = 0.4,
    ax: Axes,
    color: str = "C0",
    **kw,
) -> Axes:
    for exon in exons:
        rect = patches.Rectangle(
            (exon.start, y),
            exon.end - exon.start,
            height,
            color=color,
            **kw,
        )
        ax.add_patch(rect)
    return ax


def _plot_annotation(
    *,
    exons: List[Exon],
    height: float = 1.0,
    ax: Axes,
    color: str = "black",
) -> Axes:
    if not exons:
        return ax
    # 1. plot the exons
    _plot_transcript(
        exons=exons,
        y=0,
        height=height,
        ax=ax,
        color=color,
        zorder=0,
    )
    # 2. plot the line
    ax.hlines(
        y=height / 2,
        xmin=exons[0].start,
        xmax=exons[-1].end,
        color=color,
        lw=1.0,
        zorder=-2,
    )
    return ax


COL_OPTIONS = Optional[
    Literal[
        "celltype", "dataset_id", "technology_name", "is_long_read", "is_single_cell"
    ]
]


def plot_exons(
    data: IsoformRecord | Iterable[IsoformRecord],
    # Allow customizing colors and overall proportions only
    annotation_color: str = "black",
    color: Union[str, Tuple[int, int, int], Tuple[float, float, float]] = (
        0.21044753832183283,
        0.6773105080456748,
        0.6433941168468681,
    ),
    palette: Optional[Iterable[str]] = None,
    height_ratios: Tuple[float, float] = (1.5, 20),
    color_by: Optional[COL_OPTIONS] = "celltype",
    group_by: COL_OPTIONS = "dataset_id",
    dark_mode=True,
    dark_bg_color="#201b10ff",
    plot_junctions: bool = True,
    junction_height_ratio=1 / 3,
    height: float | int = 4,
    aspect: float | int = 3,
) -> Axes:
    """Plot exon structures and isoform abundances for a gene.

    Returns the main (isoform) axes.
    """

    # Currently, a paired axes layout (annotation over isoforms) is always created.
    # The `ax` parameter is accepted for API symmetry but not used to inject axes.
    fig, (annotation_ax, main_ax) = plt.subplots(
        2,
        1,
        figsize=(height * aspect, height),
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0.1},
        sharex=True,
    )

    # Apply dark mode styling if enabled
    if dark_mode:
        fig.patch.set_facecolor(dark_bg_color)
        annotation_ax.set_facecolor(dark_bg_color)
        main_ax.set_facecolor(dark_bg_color)
        # Update default text colors for dark mode
        text_color = "white"
        annotation_color = "white" if annotation_color == "black" else annotation_color
    else:
        text_color = "black"

    if isinstance(data, IsoformRecord):
        data = [data]
    else:
        data = list(data)
        if not all(d.gene_id == data[0].gene_id for d in data):
            raise ValueError("All IsoformRecords must have the same gene_id")

    annot_record = data[0]
    annotation_exons = annot_record.to_relative_exons()
    _plot_annotation(
        exons=annotation_exons,
        ax=annotation_ax,
        color=annotation_color,
        height=1.0,
    )

    def _get_celltype_label(celltype_dict: Dict[str, float]) -> str:
        if not celltype_dict:
            return "N/A"
        if len(celltype_dict) == 1:
            return next(iter(celltype_dict.keys()))
        items = sorted(celltype_dict.items(), key=lambda x: x[1], reverse=True)
        return ", ".join(f"{k} ({v:.0%})" for k, v in items)

    def _idx_key(key: COL_OPTIONS, record: IsoformRecord):
        if key is None:
            return None
        elif key == "celltype":
            return _get_celltype_label(record.celltype)
        else:
            return getattr(record, key)

    color_using = partial(_idx_key, color_by)
    group_using = partial(_idx_key, group_by)

    gene_body_end = max(exon.end for exon in annotation_exons)
    # Transcripts (isoform tracks)
    groups, transcripts = tuple(
        zip(
            *[
                ((g, c), t)
                for g, c, record in zip(
                    map(group_using, data), map(color_using, data), data
                )
                for t in record.to_relative_transcripts()
                if (
                    any(
                        exon.start < gene_body_end and exon.end >= 0 for exon in t.exons
                    )
                    and t.weight > 0.0
                )
            ]
        )
    )
    grouped_ordering = _get_transcripts_order(transcripts)
    order_rank = argsort(grouped_ordering)
    optimal_order = [
        (g, t)
        for g, _, t in sorted(
            zip(groups, order_rank, transcripts), key=lambda x: (x[0], x[1])
        )
    ]

    grouped_transcripts = {
        g: [t for _, t in group]
        for g, group in groupby(optimal_order, key=lambda x: x[0])
    }

    color_ids = set(c for _, c in groups)
    n_colors = max(1, len(color_ids))

    if palette is not None:
        palette_list: List[str] = [mcolors.to_hex(c) for c in list(palette)]
    elif len(data) > 1:
        # Try rcParams color cycle first
        try:
            from seaborn import husl_palette

            raw_colors = husl_palette(n_colors)
        except Exception:
            raw_colors = []
        if not raw_colors:
            # Fallback to colormap sampling from the husl color space
            cmap = plt.cm.get_cmap("husl", n_colors)
            n = getattr(cmap, "N", 10)
            raw_colors = [cmap(i % n) for i in range(n)]
        palette_list = [mcolors.to_hex(c) for c in raw_colors]
    else:
        palette_list = [mcolors.to_hex(color)]

    color_map = {
        cid: palette_list[i % len(palette_list)]
        for i, cid in enumerate(sorted(color_ids))
    }

    curr_y = 1.0
    last_group_id = None
    for (group_id, color_id), transcripts in grouped_transcripts.items():
        # Add a separator line between groups and a small label off to the side
        if group_by is not None and group_id != last_group_id:
            main_ax.axhline(y=curr_y, color="lightgrey", lw=0.35, ls="--", zorder=0)
            main_ax.text(
                x=gene_body_end * 1.01,
                y=curr_y,
                s=str(group_id),
                va="bottom",
                ha="left",
                fontsize=9,
                color="grey" if not dark_mode else "lightgrey",
                zorder=2,
            )
            last_group_id = group_id

        for transcript in transcripts:
            _plot_transcript(
                ax=main_ax,
                exons=transcript.exons,
                y=curr_y,
                color=color_map[color_id],
                height=transcript.weight,
                zorder=1,
            )
            curr_y += transcript.weight

        if plot_junctions:
            exon_height = sum(t.weight for t in transcripts)
            junctions = JunctionRecord.accumulate_junctions(
                [t.to_junctions() for t in transcripts]
            )
            _plot_junctions(
                ax=main_ax,
                junctions=junctions,
                color=color_map[color_id],
                max_arc_height=exon_height * junction_height_ratio,
                min_arc_height=exon_height * junction_height_ratio * 1 / 8,
                linewidth=0.75,
                alpha=0.5,
                zorder=1,
                height_offset=curr_y,  # position arcs above current content
            )
            curr_y += (
                exon_height * junction_height_ratio + exon_height * 1 / 10
            )  # leave some space after junctions

    if group_by is not None:
        main_ax.text(
            x=gene_body_end * 1.01,
            y=0.0,
            s=f"(\u2191{group_by.replace("_", " ").title()})",
            va="top",
            ha="left",
            fontsize=9,
            color="grey" if not dark_mode else "lightgrey",
            zorder=2,
        )

    _plot_transcript(
        ax=main_ax,
        exons=annotation_exons,
        y=0.0,
        color="white" if dark_mode else "black",
        alpha=0.05,  # if not dark_mode else 0.1,
        height=curr_y,
        zorder=-1,
    )

    if annot_record.gene_id is not None:
        annotation_ax.set_title(
            f"Gene: {annot_record.gene_id}"
            + (f" ({annot_record.gene_name})" if annot_record.gene_name else "")
            + " \u2192",
            fontsize=9,
            pad=10,
            color=text_color,
        )
    main_ax.set_ylim(-1, curr_y + 0.1)
    main_ax.set_xlim(-0.01 * gene_body_end, 1.01 * gene_body_end)
    main_ax.set_yticks([])
    main_ax.set_ylabel("Relative isoform\nabundance", fontsize=9, color=text_color)
    main_ax.set_xticks([])
    for spine in main_ax.spines.values():
        spine.set_visible(False)
    annotation_ax.axis("off")

    # Add scalebar to show gene body size
    _add_scalebar(annotation_ax, gene_body_end, y_position=0.05, color=text_color)

    # Create circular markers instead of rectangular patches
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w" if not dark_mode else dark_bg_color,
            markerfacecolor=color,
            markersize=6,
            label=color_id.title() if color_id is not None else "N/A",
            markeredgewidth=0,
        )
        for color_id, color in color_map.items()
    ]
    fig = plt.gcf()

    # Create legend with improved styling
    legend = fig.legend(
        handles=handles[::-1],
        loc="lower center",
        fontsize=9,
        ncol=min(5, len(handles)),
        bbox_to_anchor=(0.5, -0.05),
        title=color_by.replace("_", " ").title() if color_by is not None else "color",
        title_fontsize=9,
        frameon=False,  # Remove frame
        columnspacing=1.5,  # Adjust spacing between columns
        handletextpad=0.2,  # Reduce space between marker and text
    )
    for text in legend.get_texts():
        text.set_color(text_color)

    # Style the legend title
    legend.get_title().set_color(text_color)

    return main_ax
