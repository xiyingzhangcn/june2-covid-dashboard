import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import chisquare

PROCESSED = "processed"
WAVE_START, WAVE_END = "2020-02-01", "2020-08-01"
AGE_ORDER = ["0-17", "18-39", "40-59", "60-74", "75+"]


# ----------------------------------------------------------------------------
# Chi-square goodness-of-fit test on the age distribution of deaths
# ----------------------------------------------------------------------------
def age_distribution(path, date_col):
    """Total first-wave deaths per age band, in fixed order."""
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df[(df[date_col] >= WAVE_START) & (df[date_col] <= WAVE_END)]
    totals = (df.groupby("group")["deaths"].sum()
                .reindex(AGE_ORDER).fillna(0))
    return totals


def chi_square_age(run_name, real_age_totals):
    """
    Chi-square test comparing a run's age distribution of deaths with the
    real distribution.

    The test compares observed counts (simulation) against expected counts
    derived from the real distribution, scaled to the simulation's total so
    that the two have the same number of deaths. This isolates differences in
    the SHAPE of the distribution from differences in overall scale.

        chi2 = sum( (observed - expected)^2 / expected )

    A small p-value (< 0.05) indicates the simulated age distribution differs
    significantly in shape from the real one.
    """
    sim_path = f"{PROCESSED}/sim_deaths_by_age_{run_name}.csv"
    if not os.path.exists(sim_path):
        return None
    observed = age_distribution(sim_path, "week").values.astype(float)
    real = real_age_totals.values.astype(float)

    if observed.sum() == 0 or real.sum() == 0:
        return None

    # expected = real distribution rescaled to the simulation's total
    expected = real / real.sum() * observed.sum()

    # chi-square requires all expected counts > 0; drop empty bands
    mask = expected > 0
    chi2, p = chisquare(f_obs=observed[mask], f_exp=expected[mask])
    return {"run": run_name,
            "chi2": round(float(chi2), 2),
            "p_value": round(float(p), 4),
            "significant_diff": "yes" if p < 0.05 else "no"}


# ----------------------------------------------------------------------------
# Standard error and 95% CI for a ratio (e.g. case fatality ratio)
# ----------------------------------------------------------------------------
def ratio_with_se(deaths, cases):
    """
    Case fatality ratio with its standard error and 95% confidence interval,
    treating it as a binomial proportion p = deaths / cases.

        p  = deaths / cases
        SE = sqrt( p * (1 - p) / cases )
        95% CI = p +/- 1.96 * SE

    The standard error expresses how precisely the ratio is estimated. A
    difference between two ratios is only meaningful if it is large relative
    to their standard errors.
    """
    if cases <= 0:
        return None
    p = deaths / cases
    se = np.sqrt(p * (1 - p) / cases)
    lo, hi = p - 1.96 * se, p + 1.96 * se
    return {"CFR": round(p, 5),
            "SE": round(se, 5),
            "CI95_low": round(max(lo, 0), 5),
            "CI95_high": round(hi, 5)}


def cfr_for_run(run_name):
    """Compute CFR (deaths/infections) with SE for one run over the wave."""
    path = f"{PROCESSED}/sim_daily_{run_name}.csv"
    if not os.path.exists(path):
        return None
    sim = pd.read_csv(path, parse_dates=["date"])
    sim = sim[(sim["date"] >= WAVE_START) & (sim["date"] <= WAVE_END)]
    deaths = sim["deaths"].sum()
    infections = sim["infections"].sum()
    r = ratio_with_se(deaths, infections)
    if r is None:
        return None
    return {"run": run_name, **r}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # discover runs
    files = glob.glob(f"{PROCESSED}/sim_daily_*.csv")
    run_names = sorted(os.path.basename(f)[len("sim_daily_"):-len(".csv")]
                       for f in files)
    if not run_names:
        print("No runs found.")
        return

    # --- chi-square on age distribution ---
    real_age = age_distribution(f"{PROCESSED}/real_deaths_by_age.csv", "date")
    chi_rows = [r for r in (chi_square_age(run, real_age)
                            for run in run_names) if r]
    chi_table = pd.DataFrame(chi_rows)

    print("Chi-square test: age distribution of deaths (sim vs real)\n")
    print(chi_table.to_string(index=False))
    print()

    # --- CFR with standard error ---
    cfr_rows = [r for r in (cfr_for_run(run) for run in run_names) if r]
    cfr_table = pd.DataFrame(cfr_rows)

    print("Case fatality ratio with standard error (per run)\n")
    print(cfr_table.to_string(index=False))

    chi_table.to_csv(f"{PROCESSED}/chi_square_age.csv", index=False)
    cfr_table.to_csv(f"{PROCESSED}/cfr_with_se.csv", index=False)


if __name__ == "__main__":
    main()
