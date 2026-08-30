import os
import io
import numpy as np
import pandas as pd
import requests
import h5py

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RUNS = {
    "m0.4": r"full_gb_runs/full_gb_m0.4/simulation_events.h5",
    "m0.8": r"full_gb_runs/full_gb_m0.8/simulation_events.h5",
    "m1.0": r"full_gb_runs/full_gb_m1.0/simulation_events.h5",
    #"m1.2": r"full_gb_runs/full_gb_m1.2/simulation_events.h5",
}


with h5py.File(r"full_gb_runs/full_gb_m0.4/simulation_events.h5", "r") as f:
    people = f["lookups/people"]
    print("people表的字段:", people.dtype.names)





OUTPUT_DIR = "processed"
SIM_START_DATE = pd.Timestamp("2020-02-01")  
GEOGRAPHY = "England"                          

AGE_BINS = [0, 18, 40, 60, 75, 200]
AGE_LABELS = ["0-17", "18-39", "40-59", "60-74", "75+"]

UKHSA_METRICS = {
    "cases": "COVID-19_cases_casesByDay",
    "deaths": "COVID-19_deaths_ONSByDay",
}

ONS_AGESEX_URL = (
    "https://download.ons.gov.uk/downloads/datasets/"
    "weekly-deaths-age-sex/editions/covid-19/versions/37.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Load simulation events and person lookup
# ----------------------------------------------------------------------------
def load_simulation(path):
    """Read event tables and the person lookup from a JUNE 2 HDF5 file."""
    with h5py.File(path, "r") as f:
        events = {}
        for name in ["infections", "hospital_admissions",
                     "icu_admissions", "deaths"]:
            events[name] = f[f"events/{name}"][:]
        people = f["lookups/people"][:]
    return events, people


def person_attribute_lookup(people):
    """Return a function mapping person_id -> a chosen person attribute."""
    pid = people["person_id"]
    order = np.argsort(pid)
    pid_sorted = pid[order]

    def lookup(person_ids, field):
        idx = np.searchsorted(pid_sorted, person_ids)
        return people[field][order][idx]

    return lookup


# ----------------------------------------------------------------------------
# 2. Aggregate simulation events into daily time series
# ----------------------------------------------------------------------------
def daily_counts(event_time, n_days):
    days = np.floor(event_time).astype(int)
    return np.bincount(days, minlength=n_days)[:n_days]


def build_daily_table(events, n_days=182):
    """Combine the four metrics into one daily table with real dates.
    n_days is fixed (1 Feb to 1 Aug 2020 = 182 days) so all runs align."""
    table = {"day": np.arange(n_days)}
    for name, ev in events.items():
        table[name] = daily_counts(ev["time"], n_days)
    df = pd.DataFrame(table)
    df["date"] = SIM_START_DATE + pd.to_timedelta(df["day"], unit="D")
    return df[["date", "day", "infections",
               "hospital_admissions", "icu_admissions", "deaths"]]


# ----------------------------------------------------------------------------
# 3. Simulated deaths by age group and by sex (weekly)
# ----------------------------------------------------------------------------
def deaths_by_group(events, lookup, group_field, bins=None, labels=None):
    deaths = events["deaths"]
    values = lookup(deaths["person_id"], group_field)
    if group_field == "sex":
        values = np.array([v.decode() if isinstance(v, bytes) else v
                           for v in values])
    dates = SIM_START_DATE + pd.to_timedelta(np.floor(deaths["time"]), unit="D")
    df = pd.DataFrame({"date": dates, "group": values})
    if bins is not None:
        df["group"] = pd.cut(df["group"].astype(float), bins=bins,
                             labels=labels, right=False)
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    return (df.groupby(["week", "group"], observed=True)
              .size().reset_index(name="deaths"))


# ----------------------------------------------------------------------------
# 4. Real cases and deaths from the UKHSA API (England, national level)
# ----------------------------------------------------------------------------
def get_ukhsa(metric, geography_type, geography):
    url = (
        "https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease"
        "/sub_themes/respiratory/topics/COVID-19"
        f"/geography_types/{geography_type}/geographies/{geography}"
        f"/metrics/{metric}"
    )
    rows = []
    while url:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for r in data.get("results", []):
            rows.append({"date": r["date"], "value": r["metric_value"]})
        url = data.get("next")
    if not rows:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_real_daily():
    """UKHSA England cases and deaths, merged into one daily table."""
    frames = []
    for name, metric in UKHSA_METRICS.items():
        df = get_ukhsa(metric, "Nation", GEOGRAPHY)
        if df.empty:
            print(f"WARNING: UKHSA metric {metric} returned no data.")
            continue
        frames.append(df.rename(columns={"value": name}).set_index("date"))
    real = pd.concat(frames, axis=1).reset_index().sort_values("date")
    return real.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 5. Real deaths by age and sex from ONS (England and Wales, weekly)
# ----------------------------------------------------------------------------
def week_to_date(week_series, year=2020):
    n = week_series.astype(str).str.extract(r"(\d+)")[0].astype(int)
    first_friday = pd.Timestamp(f"{year}-01-03")
    return first_friday + pd.to_timedelta((n - 1) * 7, unit="D")


ONS_AGE_TO_BAND = {
    "00-01": "0-17", "01-04": "0-17", "05-09": "0-17", "10-14": "0-17",
    "15-19": "18-39",
    "20-24": "18-39", "25-29": "18-39", "30-34": "18-39", "35-39": "18-39",
    "40-44": "40-59", "45-49": "40-59", "50-54": "40-59", "55-59": "40-59",
    "60-64": "60-74", "65-69": "60-74", "70-74": "60-74",
    "75-79": "75+", "80-84": "75+", "85-89": "75+", "90+": "75+",
}


def fetch_ons_agesex_deaths():
    """Real COVID deaths by age band and by sex (England and Wales, 2020)."""
    resp = requests.get(ONS_AGESEX_URL, timeout=300)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content))
    df = df[df["Deaths"] == "Deaths involving COVID-19: registrations"].copy()
    df = df[df["calendar-years"] == 2020].copy()
    df["V4_1"] = pd.to_numeric(df["V4_1"], errors="coerce").fillna(0)
    df["date"] = week_to_date(df["Week"])

    age = df[(df["Sex"] == "All") & (df["AgeGroups"] != "All ages")].copy()
    age["group"] = age["AgeGroups"].map(ONS_AGE_TO_BAND)
    age = age.dropna(subset=["group"])
    by_age = (age.groupby(["date", "group"])["V4_1"].sum()
                 .reset_index().rename(columns={"V4_1": "deaths"}))

    sex = df[(df["AgeGroups"] != "All ages") & (df["Sex"] != "All")].copy()
    sex["group"] = sex["Sex"].str.lower()
    by_sex = (sex.groupby(["date", "group"])["V4_1"].sum()
                 .reset_index().rename(columns={"V4_1": "deaths"}))

    return by_age, by_sex


# ----------------------------------------------------------------------------
# 6. Policy timeline
# ----------------------------------------------------------------------------
def policy_timeline():
    return pd.DataFrame([
        {"date": "2020-03-23", "policy": "First national lockdown begins"},
        {"date": "2020-05-13", "policy": "Some restrictions eased"},
        {"date": "2020-06-01", "policy": "Phased school reopening"},
        {"date": "2020-06-15", "policy": "Non-essential shops reopen"},
    ])


# ----------------------------------------------------------------------------
# Process one simulation run
# ----------------------------------------------------------------------------
def process_run(run_name, sim_path):
    """Read one run and write its simulation CSVs, tagged with the run name."""
    print(f"Processing run {run_name} from {sim_path} ...")
    events, people = load_simulation(sim_path)
    lookup = person_attribute_lookup(people)

    build_daily_table(events).to_csv(
        f"{OUTPUT_DIR}/sim_daily_{run_name}.csv", index=False)
    deaths_by_group(events, lookup, "age", bins=AGE_BINS, labels=AGE_LABELS) \
        .to_csv(f"{OUTPUT_DIR}/sim_deaths_by_age_{run_name}.csv", index=False)
    deaths_by_group(events, lookup, "sex") \
        .to_csv(f"{OUTPUT_DIR}/sim_deaths_by_sex_{run_name}.csv", index=False)
    print(f"  run {run_name} done.")


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main():
    # --- simulation side: process every run ---
    for run_name, sim_path in RUNS.items():
        if os.path.exists(sim_path):
            process_run(run_name, sim_path)
        else:
            print(f"WARNING: run file not found, skipping: {sim_path}")

    # --- real data: downloaded once, shared by all runs ---
    real_daily = build_real_daily()
    real_daily.to_csv(f"{OUTPUT_DIR}/real_daily.csv", index=False)

    real_age, real_sex = fetch_ons_agesex_deaths()
    real_age.to_csv(f"{OUTPUT_DIR}/real_deaths_by_age.csv", index=False)
    real_sex.to_csv(f"{OUTPUT_DIR}/real_deaths_by_sex.csv", index=False)

    # --- policy timeline ---
    policy_timeline().to_csv(f"{OUTPUT_DIR}/policy_timeline.csv", index=False)

    print("Processing complete. Files written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
