# HRV-Analysis-Dashboard-with-Dash-Plotly

Single-page Dashboard to analyze HRV Indices between Normal and CHF Subjects

Quick Dashboard to analyze short-term HRV indices extracted from healthy people and congestive heart failure patients

Data extracted from:

- Healthy subjects:
  - [NSRDB](!https://www.physionet.org/content/nsrdb/1.0.0/)
  - [NSR2DB](!https://www.physionet.org/content/nsr2db/1.0.0/)
- CHF patients:
  - [CHFDB](!https://www.physionet.org/content/chf2db/1.0.0/)
  - [CHF2DB](!https://www.physionet.org/content/chf2db/1.0.0/)

Data preprocessing:

- Records that contain unknown age or gender information were dropped

- Segmentation methods used to compute short-term HRV are from [this paper](!https://doi.org/10.1016/j.bspc.2019.101583)

- Missing values and ectopic beats were handled with linear interpolation method

- HRV metrics were computed using [this library](!https://github.com/Aura-healthcare/hrv-analysis)