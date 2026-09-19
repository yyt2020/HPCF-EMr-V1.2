# Data and Code Availability

## Data availability

The strong-motion records used in this study were provided by the China
Earthquake Networks Center (CENC) and are subject to the Center's data-access
policy. Because the authors do not hold redistribution rights, the original
CENC records are not included in this repository. Researchers may request
access through the National Earthquake Science Data Center
(<https://data.earthquake.cn/>) in accordance with the Center's applicable
access conditions.

PEER NGA-West3 data used in the study are publicly accessible through the
NGA-gm data service (<https://nga-gm.org/>) and the official NGA-West3 report
and electronic supplements
(<https://www.risksciences.ucla.edu/girs-reports/2025/07>; report DOI:
<https://doi.org/10.34948/N3D30R>).

The Guo method was implemented using the publicly available software released
by Guo et al.
(<https://github.com/gwx81/automatic-filter-cut-off-frequency-selection-process-for-strong-motion-records>).
The RS method was implemented using routines from the open-source USGS
`gmprocess` package (<https://pypi.org/project/gmprocess/>). P-wave arrival times
were determined using Kalkan's PPHASEPICKER algorithm and its associated MATLAB
implementation
(<https://www.mathworks.com/matlabcentral/fileexchange/57729-an-automated-p-phase-arrival-time-picker-with-snr-output>).

Any additional non-restricted derived files not included in the repository may
be requested from the corresponding author for research and verification;
access remains subject to restrictions applicable to the underlying CENC
records.

## Code availability

The English source code, configuration file, dependency specification, input
metadata template, and shareable processed result tables are provided in this
repository. The original code and documentation in this repository are
available under the MIT License (see `LICENSE`). No restricted CENC waveform
files are included.

For reproducible citation of a fixed version, users may refer to a tagged
GitHub release or a persistent archive of a release when one is available.
Third-party software and external implementations remain subject to their own
licenses and terms.
