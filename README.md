# HPCF Estimation from Strong-Motion Acceleration Records

This repository contains the English version of the code used to estimate
high-pass corner frequencies (HPCFs) from strong-motion acceleration records,
together with shareable processed result tables used in the associated study.

The original China Earthquake Networks Center (CENC) acceleration records are
not included because the authors do not hold redistribution rights. Authorized
users can supply records obtained under the Center's data-access policy and
reproduce the workflow locally.

The computational formulas, frequency-search logic, filtering modes, numerical
integration, EM calculation, and AIC picking sequence are unchanged from the
author-provided implementation. Language, filenames, portable paths, and the
minimum interface repairs needed to connect the three original stages were the
only code-level changes.

The original code in this repository is distributed under the MIT License; see
`LICENSE`. Third-party software and external implementations remain subject to
their own licenses and terms.

## Core workflow

1. `01_fas_corner_frequency.py` calculates the Fourier amplitude spectrum
   (FAS), applies Konno-Ohmachi smoothing, evaluates the log-log spectral slope,
   and estimates preliminary HPCF and low-pass corner frequency (LPCF) values.
2. `02_em_score.py` applies the original noncausal frequency-domain Butterworth
   high-pass filter over a record-specific frequency grid, integrates the
   filtered acceleration to displacement, and calculates adjacent-waveform
   envelope-misfit (EM) scores.
3. `03_aic_hpcf.py` evaluates the forward AIC curve of the EM slope intensity
   and reports the final EM-based HPCF.

`config.ini` is the control file. The default outer parallel process count is
10 (`N_process = 10`). Reduce this value on memory-limited systems.

## Repository contents

```text
HPCF-EMr-V1.2/
|-- LICENSE
|-- 01_fas_corner_frequency.py
|-- 02_em_score.py
|-- 03_aic_hpcf.py
|-- hpcf_filter.py
|-- run_pipeline.py
|-- config.ini
|-- requirements.txt
|-- DATA_AVAILABILITY.md
|-- input/
|   |-- README.md
|   `-- station_metadata_template.csv
`-- paper_results/
    |-- README.md
    `-- wenchuan_method_comparison_results.csv
```

The repository intentionally contains no raw waveform file. Generated runtime
output is written to `OUTPUT/` and is ignored by Git. Shareable processed
results are provided under `paper_results/`.

## Environment and dependencies

Python 3.11 is recommended. Create a clean environment because `esi-core`
contains compiled extensions and must be compatible with NumPy.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Linux or macOS, activate with `source .venv/bin/activate`.

The runtime libraries are NumPy, Pandas, SciPy, Matplotlib, ObsPy, tqdm, and
`esi-core`; tested versions are pinned in `requirements.txt`. On Windows,
building `esi-core` from source may require Microsoft C++ Build Tools. A clean
Python 3.11 environment with NumPy 1.26 is recommended.

## Input acquisition and preparation

CENC strong-motion records used in the study are controlled by the Center's
data-access policy and cannot be redistributed by the authors. Access may be
requested through the National Earthquake Science Data Center:

<https://data.earthquake.cn/>

After obtaining authorization, place the files as follows:

```text
input/
|-- station_metadata.csv
`-- acceleration_records/
    |-- record_001.dat
    `-- ...
```

Copy `input/station_metadata_template.csv` to `input/station_metadata.csv` and
fill one row per record. The `Record` values must exactly match physical
filenames. The program reads the record name from the fifth column and the
earthquake name from the seventh column, so column order must not be changed.

For the CENC text format expected here, the sampling interval is parsed from
line 12 and acceleration samples begin on line 17. See `input/README.md` for
the complete metadata schema.

PEER NGA-West3 data used in the study are publicly accessible from
the NGA-gm project (<https://nga-gm.org/>) and the official NGA-West3 report and
electronic supplements
(<https://www.risksciences.ucla.edu/girs-reports/2025/07>; report DOI:
<https://doi.org/10.34948/N3D30R>).

## Configuration and execution

The principal `config.ini` values are:

| Key | Default | Purpose |
|---|---:|---|
| `FAS_outdir` | `OUTPUT/01_fas` | Stage-1 output directory |
| `fmax` | `10` | Maximum EM frequency in Hz |
| `w0` | `15` | EM central-frequency parameter |
| `total` | `0` | Number of records; `0` processes all |
| `N_process` | `10` | Outer parallel worker count |
| `need_downsample` | `1` | Enable dynamic downsampling |
| `in_dir` | `input/acceleration_records` | User-supplied waveform directory |
| `station_metadata` | `input/station_metadata.csv` | User-supplied metadata file |

Run all stages from the repository root:

```bash
python run_pipeline.py
```

The runner checks that the user-supplied inputs exist, then executes all three
stages in the required order. Individual stages can also be run sequentially:

```bash
python 01_fas_corner_frequency.py
python 02_em_score.py
python 03_aic_hpcf.py
```

## Outputs

Stage 1 writes preliminary HPCF/LPCF values, smoothed spectra, and diagnostic
figures under `OUTPUT/01_fas/`. Stage 2 writes adjacent EM curves under
`OUTPUT/02_em_curves/`. Stage 3 writes final AIC-selected HPCF values under
`OUTPUT/03_aic_hpcf_results/`.

## Verification and computational integrity

The English package was compared with a minimally repaired copy of the
author-provided implementation. After sorting records to account for parallel
completion order, the maximum absolute difference was `0.0` at every stage:

- preliminary HPCF, event mean HPCF, and LPCF;
- frequency steps and all EM values; and
- final AIC-selected HPCF.

The necessary interface and portability repairs were limited to connecting the
actual stage-1 output fields to stage 2, normalizing paths, removing a
non-picklable no-op process-pool initializer, and replacing legacy absolute
paths with `config.ini` settings. No filtering equation, search grid, taper,
FFT, integration, EM expression, AIC expression, threshold, algorithm branch,
or stage order was changed.

Parallel workers may write rows in a different order between runs. Compare
results by record identifier rather than row position.

## Third-party method attribution

The method comparison includes implementations based on these external
resources; they are cited for methodological provenance and are not additional
runtime dependencies of this three-stage package:

- Guo et al. software:
  <https://github.com/gwx81/automatic-filter-cut-off-frequency-selection-process-for-strong-motion-records>
- USGS `gmprocess` package used for the RS method:
  <https://pypi.org/project/gmprocess/>
- Kalkan's PPHASEPICKER MATLAB implementation:
  <https://www.mathworks.com/matlabcentral/fileexchange/57729-an-automated-p-phase-arrival-time-picker-with-snr-output>

See `DATA_AVAILABILITY.md` for data-access and code-availability information.
The MIT License applies to the original code and documentation in this
repository. Third-party software, external implementations, and restricted
data remain subject to their own licenses, terms, and access conditions.
