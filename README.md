# JUNE 2 vs Real COVID-19: First Wave Evaluation (England)

This repository contains the code and processed data for a dissertation 
evaluating how well the JUNE 2 agent-based model reproduces the first wave 
of the COVID-19 pandemic in England, across a transmissibility sweep 
(m0.4, m0.8, m1.0).

## Scripts

- `data_processing.py` — reads the JUNE 2 simulation event logs (HDF5) and 
  downloads the real-world data (UKHSA cases and deaths, ONS deaths by age 
  and sex), producing clean CSV files in the `processed/` folder.
- `compare_runs.py` — computes the comparison metrics (RMSE, MAE, correlation, 
  peak differences, total ratio) for each run against the real data.
- `statistics.py` — computes the chi-square test on the age distributions and 
  the standard errors for the fatality ratios.
- `dashboard.py` — an interactive Plotly Dash dashboard presenting the 
  comparison, with a run selector and multiple views.

## Processed data included

The `processed/` folder contains the pre-computed CSV files produced by 
`data_processing.py`. Because these are included, the dashboard and the 
analysis scripts can be run directly, without access to the original 
simulation files.

## Requirements

Install the required packages: pip install h5py numpy pandas requests dash plotly scipy


## Option A: run the dashboard directly

The `processed/` data are already provided, so you can run the dashboard 
straight away: `dashboard.py`

Then open http://127.0.0.1:8050 in a web browser. The comparison scripts 
can also be run directly: `compare_runs.py` `statistics.py`


## Option B: regenerate the data from the simulation files

To reproduce the processed data from scratch, you need the JUNE 2 simulation 
files, which were provided by the project supervisor and are **not included** 
in this repository (see Data note below).

1. Place the simulation run folders somewhere on your machine, each containing 
   a `simulation_events.h5` file.
2. Open `data_processing.py` and edit the `RUNS` dictionary near the top so 
   that each path points to your `simulation_events.h5` file, for example:

```python
   RUNS = {
       "m0.4": r"C:\your\path\full_gb_m0.4\simulation_events.h5",
       "m0.8": r"C:\your\path\full_gb_m0.8\simulation_events.h5",
       "m1.0": r"C:\your\path\full_gb_m1.0\simulation_events.h5",
   }
```
3. Run the processing, which downloads the real data and writes the CSV files 
   into `processed/`
4. Then run the dashboard or the analysis scripts as in Option A.

## Data note

The real-world data are downloaded automatically from public UKHSA and ONS 
sources when `data_processing.py` is run. The JUNE 2 simulation files 
(HDF5) were provided by the project supervisor and are not included in this 
repository. The processed CSV files derived from them are included so that 
the analysis and dashboard can be run without the original files.   
