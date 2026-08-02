r"""
figstyle.py - shared visual style. Support file, imported by every figure script.

Type sizes, column widths, palette and mark specs live here so they cannot drift
between figures. Print-oriented: vector PDF at journal column widths, fonts
embedded as Type 42.

PALETTE. Two categorical hues, validated for colour-vision deficiency: worst
adjacent-pair separation dE 24.7 (protan) / 32.7 (tritan), normal-vision dE 33.6,
both above 3:1 contrast on white. Do NOT substitute without re-validating - a
red/green pair will fail deuteranopia separation, which is the most common form.

COLOUR FOLLOWS THE ENTITY, NEVER THE RANK. A condition keeps its hue in every
figure regardless of whether it wins in that panel. Where two dimensions must be
distinguished in one panel, the second uses LINESTYLE, not a third hue, so the
figure survives greyscale printing.

NO NUMBERS ON FIGURES. Values go to LaTeX tables emitted by the same script that
draws the figure, so the two can never disagree and nothing is transcribed by
hand. Reference lines may be drawn and named; their values live in the table.
"""
from __future__ import annotations

import matplotlib as mpl

# --------------------------------------------------------------- palette ----
C_A = "#2a78d6"        # categorical slot 1
C_B = "#eb6834"        # categorical slot 2
C_INK = "#0b0b0b"      # primary text / summary marks
C_MUTED = "#52514e"    # secondary text, reference lines, annotation
C_GRID = "#d9d8d4"     # recessive grid

# Semantic aliases - use these in figure scripts so the meaning is explicit and
# a palette change is a one-line edit here.
C_SIMULATION = C_A
C_BENCH = C_B
C_RANDOM = C_A
C_WARM = C_B
C_IDEAL = C_A
C_FAITHFUL = C_B

LS_SOLID = "-"
LS_DASH = (0, (4, 2))
LS_DOT = (0, (2, 2))

# Optica / Optics Express two-column geometry, inches.
COL_W = 3.35
COL_W_2 = 6.85

# The SUBMISSION text block. optica-article.cls sets \textwidth = \columnwidth
# = 379.4pt = 5.25in: the submission format is single column, not two. Any figure
# authored wider than this and placed with width=\linewidth is scaled DOWN, and
# its type shrinks with it.
TEXT_W = 5.25


def title_size(authored_width, target_pt=8.0):
    r"""Font size that lands at target_pt once the figure is scaled to TEXT_W.

    A 9in panel placed at \linewidth is scaled 0.583x, so a 6.5pt title reaches
    the page at 3.8pt - far under Optica's 7-9pt guidance, and the reason the
    reconstruction panels read as small. Deriving the size from the authored
    width makes every layout land at the same size on the page no matter how
    wide it was drawn. Do not hardcode a font size in a figure that gets scaled.
    """
    return target_pt * max(float(authored_width), 1e-6) / TEXT_W

def apply() -> None:
    """Install the manuscript rcParams. Call once per figure script."""
    mpl.rcParams.update({
        # SERIF, to match the manuscript. optica-article.cls sets the body in
        # TeX Gyre Termes via newtx; sans-serif figure labels made the Delta and
        # the minus sign in an axis label render in a visibly different face from
        # the same symbols in the caption directly beneath it.
        #
        # The fallback chain is deliberate. Times New Roman resolves on Windows;
        # TeX Gyre Termes is what newtx itself uses and ships with TeX Live;
        # STIXGeneral is bundled WITH matplotlib, so the chain cannot fail
        # silently and fall back to sans. Do not shorten it.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TeX Gyre Termes", "STIXGeneral",
                       "DejaVu Serif"],
        # Math in the same face. "stix" is bundled with matplotlib and is
        # Times-metric-compatible, so $\Delta$ in a figure matches $\Delta$ in a
        # caption. Do NOT use fontset="custom" - that depends on a named system
        # font being discoverable and degrades with warnings if it is not.
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        "axes.edgecolor": C_MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": C_INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.5,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,

        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "lines.solid_capstyle": "round",

        "legend.frameon": False,
        "legend.handlelength": 1.8,

        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, stem: str) -> None:
    """PDF for LaTeX, PNG for quick inspection in the notebook."""
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=200)
    print(f"  figure -> {stem}.pdf / .png")
