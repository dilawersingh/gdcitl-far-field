# Camera-in-the-loop gradient descent hologram generation for Fraunhofer diffraction

Method and analysis code for:

> D. Singh, A. J. Wojcik and T. D. Wilkinson, "Camera-in-the-loop gradient
> descent hologram generation for Fraunhofer diffraction" (under consideration).

**This repository holds the code only.** The experimental record, the simulation
record and the scored results are deposited in the University of Cambridge Apollo
repository:

> D. Singh, A. J. Wojcik and T. D. Wilkinson, "Data and analysis code for
> 'Camera-in-the-loop gradient descent hologram generation for Fraunhofer
> diffraction'", University of Cambridge Apollo Repository (2026).
> https://doi.org/10.17863/CAM.132918

The deposit is about 3.9 GB, which is why it is not here. It contains this same
code alongside the data, so if you want to reproduce the paper, download the
deposit rather than cloning this. This repository exists so the code can be read,
diffed and linked to without downloading gigabytes of camera captures.

The deposit becomes publicly available on publication.

## What is here

| path | what it is |
|---|---|
| `code/method/` | The method itself: forward model, gradient descent, capture processing, scoring. |
| `code/analysis/` | Produces every figure and table in the paper. Reads the results tree, writes PDFs and `.tex`. |
| `code/analysis/paths.py` | Resolves every location relative to the deposit root. Nothing else hard-codes a path. |
| `code/configurations/` | The launcher configuration for each of the three experiments. |
| `code/notebooks/` | The two notebooks that were run. |
| `requirements.txt` | Direct dependencies, lower bounds not pins. |

Not here, because they only make sense with the data: `results/`, `output/`,
`targets/`, and `verify_deposit.py`, which checks a deposit tree that a bare
clone does not have.

## Running it against the data

`paths.py` finds the deposit root two levels above its own file, and everything
else resolves from there. Point it at a downloaded deposit:

```powershell
$env:GDCITL_ROOT = "C:\path\to\gdcitl-far-field"
```

```bash
export GDCITL_ROOT=/path/to/gdcitl-far-field
```

Then `python code/analysis/paths.py` prints every location and whether it exists,
and `code/notebooks/gd_citl_analysis.ipynb` regenerates every figure, table and
reported number.

`code/notebooks/gd_citl_experiments.ipynb` is the acquisition notebook. It is a
record of the procedure, not something you can re-run: it needs the spatial light
modulator, the camera, and the manufacturers' SDKs, which cannot be
redistributed.

## A note on naming

The paper calls one condition *simulation-seeded*. On disk it is `warm_start`,
in the source identifier `bench_citl_warm_start`, the directory `warm_start/` and
the configuration value `citl_init_phase: "warm_start"`. Those are the literal
strings the code and the recorded runs use, so they are left alone rather than
rewritten after the fact. Every label a reader sees in a figure or table says
simulation-seeded.

The bench result directories carry a `citl_gs_gd_` prefix from the launcher in a
shared codebase. Only gradient descent is reported in this work.

## License

MIT. See `LICENSE`.

The data in the Apollo deposit is released separately under CC BY 4.0, and the
target images it contains are third-party material under their own terms. See
the deposit's `LICENSE-data.txt` and `targets/CREDITS.md`.

## Contact

Dilawer Singh, Department of Engineering, University of Cambridge <br>
Email: ds2070@cam.ac.uk
