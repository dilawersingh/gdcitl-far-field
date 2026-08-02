r"""
paths.py - where everything lives, resolved from this file's own location.

WHY THIS EXISTS.

  This deposit is self-contained: the targets, the results and the configuration
  are all inside it. So nothing should need editing before it runs. Every path is
  resolved relative to the deposit root, which is found from this file's location
  rather than from the working directory - so it behaves the same whether you run
  a script from the repository root, from code/analysis/, or from a notebook
  opened somewhere else entirely.

  Set GDCITL_ROOT to point the whole thing at a different tree (for example, at
  your own results while re-running the experiments). Everything else follows.

USAGE
    import paths
    print(paths.TARGETS)          # <deposit>/targets
    print(paths.RESULTS)          # <deposit>/results
    print(paths.SCORED)           # <deposit>/output/scored.json
    paths.report()                # print every location and whether it exists
"""
from __future__ import annotations

import os
from pathlib import Path

# code/analysis/paths.py -> code/analysis -> code -> <deposit root>
_HERE = Path(__file__).resolve()
ROOT = Path(os.environ.get("GDCITL_ROOT", _HERE.parents[2]))

TARGETS = ROOT / "targets"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "code" / "configurations"

# roi.json is an INPUT - the magnified-region coordinates, chosen by hand from
# the targets before any reconstruction was viewed. It lives with the results
# because it describes them; it is not produced by anything here.
ROI = RESULTS / "roi.json"
# These three named a layout the deposit never had: report() printed them as
# missing and the README described them. They now name what is actually there.
ZERNIKE = RESULTS / "simulations" / "zernike_stress_test"
EXPERIMENTS = RESULTS / "experiments"
SIMULATIONS = RESULTS / "simulations"
REPLAY = RESULTS / "experiments" / "replay_simulation"

# WHAT THE ANALYSIS PRODUCES, all under one root.
#
# results/ holds what the EXPERIMENTS produced - captures, run logs, simulation
# sweeps. output/ holds what the ANALYSIS produced from them. scored.json belongs
# here rather than in results/ for the same reason the figures do: it is derived,
# and the README says so. Keeping the two apart means a rebuild can never write
# into the deposit's own record of what was run.
#
# Note the asymmetry when clearing for a clean rebuild: deleting output/ removes
# scored.json too, and the analysis notebook then has no reference to diff its
# rebuild against. It degrades gracefully - it says so and carries on - but
# delete output/figures and output/tables rather than output/ if you want the
# comparison.
OUTPUT = ROOT / "output"
SCORED = OUTPUT / "scored.json"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"


def resolve(value):
    """Turn a config path value into an absolute path under ROOT.

    Accepts a string or the single-element list the configs use. Relative values
    are taken as relative to the deposit root; absolute values are left alone, so
    pointing a config at an external results tree still works.
    """
    if isinstance(value, (list, tuple)):
        value = value[0]
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p)


def report() -> None:
    print(f"  ROOT      {ROOT}"
          f"{'' if ROOT.exists() else '   ** MISSING **'}")
    for name in ("TARGETS", "RESULTS", "CONFIGS", "ROI",
                 "EXPERIMENTS", "SIMULATIONS", "ZERNIKE", "REPLAY",
                 "OUTPUT", "SCORED", "FIGURES", "TABLES"):
        p = globals()[name]
        mark = "" if p.exists() else "   ** missing **"
        print(f"  {name:9s} {p}{mark}")


if __name__ == "__main__":
    report()