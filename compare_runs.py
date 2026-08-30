import glob
import os
import numpy as np
import pandas as pd

PROCESSED = "processed"
WAVE_START, WAVE_END = "2020-02-01", "2020-08-01"


# ----------------------------------------------------------------------------
# Load and align one run's weekly deaths with real weekly deaths
# ----------------------------------------------------------------------------
def weekly_deaths_from_daily(df):
    """Aggregate a daily death series to weekly totals (week-start index)."""
    d = df.copy()
    d["week"] = d["date"].dt.to_period("W").dt.start_time
    return d.groupby("week")["deaths"].sum().reset_index()


def load_real_weekly():
    real = pd.read_csv(f"{PROCESSED}/real_daily.csv", parse_dates=["date"])
    real = real[(real["date"] >= WAVE_START) & (real["date"] <= WAVE_END)]
    return weekly_deaths_from_daily(real).rename(
        columns={"deaths": "real"})


def load_run_weekly(run_name):
    path = f"{PROCESSED}/sim_daily_{run_name}.csv"
    sim = pd.read_csv(path, parse_dates=["date"])
    sim = sim[(sim["date"] >= WAVE_START) & (sim["date"] <= WAVE_END)]
    return weekly_deaths_from_daily(sim).rename(columns={"deaths": "sim"})


# ----------------------------------------------------------------------------
# The comparison metrics (the "equations")
# ----------------------------------------------------------------------------
def rmse(sim, real):
    """Root mean square error: sqrt(mean((sim - real)^2))."""
    return float(np.sqrt(np.mean((sim - real) ** 2)))


def mae(sim, real):
    """Mean absolute error: mean(|sim - real|)."""
    return float(np.mean(np.abs(sim - real)))


def pearson_r(sim, real):
    """Pearson correlation coefficient between the two series."""
    if np.std(sim) == 0 or np.std(real) == 0:
        return float("nan")
    return float(np.corrcoef(sim, real)[0, 1])


def peak_differences(weeks, sim, real):
    """Difference in peak timing (weeks) and peak height (deaths)."""
    i_sim, i_real = int(np.argmax(sim)), int(np.argmax(real))
    dt_weeks = (pd.Timestamp(weeks[i_sim]) - pd.Timestamp(weeks[i_real])).days / 7
    dy = float(sim[i_sim] - real[i_real])
    return dt_weeks, dy


def total_ratio(sim, real):
    """Ratio of total simulated deaths to total real deaths over the wave."""
    return float(np.sum(sim) / np.sum(real)) if np.sum(real) > 0 else float("nan")


# ----------------------------------------------------------------------------
# Compare one run against real
# ----------------------------------------------------------------------------
def compare_one(run_name, real_weekly):
    run_weekly = load_run_weekly(run_name)
    merged = pd.merge(run_weekly, real_weekly, on="week", how="inner")
    if len(merged) < 3:
        return None
    sim = merged["sim"].values.astype(float)
    real = merged["real"].values.astype(float)
    weeks = merged["week"].values
    dt_peak, dy_peak = peak_differences(weeks, sim, real)
    return {
        "run": run_name,
        "weeks_compared": len(merged),
        "RMSE": round(rmse(sim, real), 1),
        "MAE": round(mae(sim, real), 1),
        "correlation": round(pearson_r(sim, real), 3),
        "peak_timing_diff_weeks": round(dt_peak, 1),
        "peak_size_diff": round(dy_peak, 0),
        "total_ratio_sim_over_real": round(total_ratio(sim, real), 3),
    }


# ----------------------------------------------------------------------------
# Main: find all runs, compare each, rank them
# ----------------------------------------------------------------------------
def main():
    real_weekly = load_real_weekly()

    # discover runs from the sim_daily_*.csv files present
    files = glob.glob(f"{PROCESSED}/sim_daily_*.csv")
    run_names = sorted(os.path.basename(f)[len("sim_daily_"):-len(".csv")]
                       for f in files)
    if not run_names:
        print("No sim_daily_*.csv files found in", PROCESSED)
        return

    rows = []
    for run in run_names:
        result = compare_one(run, real_weekly)
        if result:
            rows.append(result)

    table = pd.DataFrame(rows)
    table.to_csv(f"{PROCESSED}/run_comparison.csv", index=False)

    print("Run comparison (weekly deaths, England, first wave):\n")
    print(table.to_string(index=False))
    print()

    # A simple summary of which run is closest on each metric.
    if not table.empty:
        best_rmse = table.loc[table["RMSE"].idxmin(), "run"]
        best_corr = table.loc[table["correlation"].idxmax(), "run"]
        best_ratio = table.loc[
            (table["total_ratio_sim_over_real"] - 1).abs().idxmin(), "run"]
        print(f"Lowest RMSE:            {best_rmse}")
        print(f"Highest correlation:    {best_corr}")
        print(f"Total ratio closest 1:  {best_ratio}")


if __name__ == "__main__":
    main()
