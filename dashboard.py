import glob
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

PROCESSED = "processed"
WAVE_START, WAVE_END = "2020-02-01", "2020-08-01"
AGE_ORDER = ["0-17", "18-39", "40-59", "60-74", "75+"]
SEX_ORDER = ["female", "male"]
SIM_COLOR = "steelblue"
REAL_COLOR = "firebrick"


def discover_runs():
    files = glob.glob(f"{PROCESSED}/sim_daily_*.csv")
    return sorted(os.path.basename(f)[len("sim_daily_"):-len(".csv")]
                  for f in files)

RUNS = discover_runs()

real_daily = pd.read_csv(f"{PROCESSED}/real_daily.csv", parse_dates=["date"])
real_age = pd.read_csv(f"{PROCESSED}/real_deaths_by_age.csv", parse_dates=["date"])
real_sex = pd.read_csv(f"{PROCESSED}/real_deaths_by_sex.csv", parse_dates=["date"])
policies = pd.read_csv(f"{PROCESSED}/policy_timeline.csv", parse_dates=["date"])


def load_run_daily(run):
    return pd.read_csv(f"{PROCESSED}/sim_daily_{run}.csv", parse_dates=["date"])

def load_run_age(run):
    return pd.read_csv(f"{PROCESSED}/sim_deaths_by_age_{run}.csv",
                       parse_dates=["week"])

def load_run_sex(run):
    return pd.read_csv(f"{PROCESSED}/sim_deaths_by_sex_{run}.csv",
                       parse_dates=["week"])


# ---------------------------------------------------------------------------
# Policy lines with labels on the chart
# ---------------------------------------------------------------------------
def add_policy_lines(fig, labelled=True):
    for _, row in policies.iterrows():
        fig.add_vline(x=row["date"], line_width=1, line_dash="dot",
                      line_color="grey")
        if labelled:
            fig.add_annotation(
                x=row["date"], yref="paper", y=1.0,
                text=row["policy"], showarrow=False,
                textangle=-90, xshift=-7, yshift=-2,
                font=dict(size=8, color="grey"),
                xanchor="right", yanchor="top")
    return fig


def wave(df, col="date"):
    return df[(df[col] >= WAVE_START) & (df[col] <= WAVE_END)]

def weekly_deaths(df, datecol):
    d = df.copy()
    d["week"] = d[datecol].dt.to_period("W").dt.start_time
    return d.groupby("week")["deaths"].sum().reset_index()

def group_proportions(df, datecol, order):
    w = df[(df[datecol] >= WAVE_START) & (df[datecol] <= WAVE_END)]
    totals = w.groupby("group", observed=True)["deaths"].sum().reindex(
        order).fillna(0)
    tot = totals.sum()
    return (totals / tot * 100) if tot > 0 else totals


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def overview_figure(run, metric):
    sim = load_run_daily(run)
    if metric == "cases":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=sim["date"], y=sim["infections"],
                                 name="Simulation", line=dict(color=SIM_COLOR)),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=real_daily["date"], y=real_daily["cases"],
                                 name="Real", line=dict(color=REAL_COLOR)),
                      secondary_y=True)
        fig.update_yaxes(title_text="Simulated daily infections",
                         secondary_y=False)
        fig.update_yaxes(title_text="Real daily cases", secondary_y=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim["date"], y=sim["deaths"],
                                 name="Simulation", line=dict(color=SIM_COLOR)))
        fig.add_trace(go.Scatter(x=real_daily["date"], y=real_daily["deaths"],
                                 name="Real", line=dict(color=REAL_COLOR)))
        fig.update_yaxes(title_text="Daily deaths")
    add_policy_lines(fig)
    fig.update_layout(
        title=f"{metric.capitalize()}: run {run} vs real data",
        xaxis_title="Date", xaxis_range=[WAVE_START, WAVE_END],
        template="simple_white", hovermode="x unified", margin=dict(t=110),
        legend=dict(orientation="h", yanchor="bottom", y=1.10, x=0))
    return fig


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------
def demographics_distribution(run, dimension):
    if dimension == "age":
        order = AGE_ORDER
        sim_pct = group_proportions(load_run_age(run), "week", order)
        real_pct = group_proportions(real_age, "date", order)
        axis = "Age group"
    else:
        order = SEX_ORDER
        sim_pct = group_proportions(load_run_sex(run), "week", order)
        real_pct = group_proportions(real_sex, "date", order)
        axis = "Sex"
    labels = [str(g) for g in order]
    ymax = max(max(sim_pct.values, default=0),
               max(real_pct.values, default=0))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=sim_pct.values,
                         name=f"Simulation ({run})", marker_color=SIM_COLOR,
                         text=[f"{v:.1f}%" for v in sim_pct.values],
                         textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=real_pct.values,
                         name="Real (England & Wales)", marker_color=REAL_COLOR,
                         text=[f"{v:.1f}%" for v in real_pct.values],
                         textposition="outside"))
    fig.update_layout(
        title=f"First-wave deaths by {dimension}: run {run} vs real",
        xaxis_title=axis, yaxis_title="Share of first-wave deaths (%)",
        yaxis_range=[0, ymax * 1.18], barmode="group",
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def demographics_over_time(run, dimension, group):
    if dimension == "age":
        sim_df, real_df, order = load_run_age(run), real_age, AGE_ORDER
    else:
        sim_df, real_df, order = load_run_sex(run), real_sex, SEX_ORDER
    if group not in order:
        group = order[-1]

    def norm(df, datecol):
        sub = df[(df[datecol] >= WAVE_START) & (df[datecol] <= WAVE_END)
                 & (df["group"] == group)].sort_values(datecol)
        y = sub["deaths"].astype(float).values
        if len(y) and y.max() > 0:
            y = y / y.max() * 100
        return sub[datecol].values, y

    sx, sy = norm(sim_df, "week")
    rx, ry = norm(real_df, "date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name="Simulation",
                             line=dict(color=SIM_COLOR)))
    fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines+markers", name="Real",
                             line=dict(color=REAL_COLOR)))
    add_policy_lines(fig)
    fig.update_layout(
        title=f"{group}: simulation vs real deaths over time "
              f"(each scaled to its own peak)",
        xaxis_title="Date", yaxis_title="Share of own peak (%)",
        xaxis_range=[WAVE_START, WAVE_END],
        template="simple_white", hovermode="x unified", margin=dict(t=110),
        legend=dict(orientation="h", yanchor="bottom", y=1.10, x=0))
    return fig


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_at_shift(run, shift_weeks):
    sim_w = weekly_deaths(wave(load_run_daily(run)), "date").rename(
        columns={"deaths": "sim"})
    sim_w["week"] = sim_w["week"] - pd.to_timedelta(shift_weeks * 7, unit="D")
    real_w = weekly_deaths(wave(real_daily), "date").rename(
        columns={"deaths": "real"})
    m = pd.merge(sim_w, real_w, on="week", how="inner")
    if len(m) < 3:
        return None
    s, r = m["sim"].values.astype(float), m["real"].values.astype(float)
    rmse = float(np.sqrt(np.mean((s - r) ** 2)))
    corr = float(np.corrcoef(s, r)[0, 1]) if s.std() and r.std() else np.nan
    return {"n": len(m), "rmse": rmse, "corr": corr, "merged": m}

def fit_figure(run, shift):
    f = fit_at_shift(run, shift)
    fig = go.Figure()
    if f:
        m = f["merged"]
        fig.add_trace(go.Scatter(x=m["week"], y=m["sim"],
                                 name=f"Simulation (shift {shift}w)",
                                 line=dict(color=SIM_COLOR)))
        fig.add_trace(go.Scatter(x=m["week"], y=m["real"], mode="lines+markers",
                                 name="Real", line=dict(color=REAL_COLOR)))
    fig.update_layout(
        title=f"Run {run}: weekly deaths, simulation shifted {shift} weeks",
        xaxis_title="Week", yaxis_title="Weekly deaths",
        template="simple_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig

def fit_metrics_line(run, shift):
    f = fit_at_shift(run, shift)
    if not f:
        return "Not enough overlapping weeks."
    return (f"At a {shift}-week shift:  correlation = {f['corr']:.3f},  "
            f"RMSE = {f['rmse']:.0f},  overlapping weeks = {f['n']}")


# ---------------------------------------------------------------------------
# Metrics: table + severity ratios (restored)
# ---------------------------------------------------------------------------
def compute_metrics(run):
    s = wave(load_run_daily(run))
    sim_cfr = s["deaths"].sum() / max(s["infections"].sum(), 1)
    rd = wave(real_daily)
    real_cfr = rd["deaths"].sum() / max(rd["cases"].sum(), 1)
    sim_peak = s.loc[s["deaths"].idxmax()]
    real_peak = rd.loc[rd["deaths"].idxmax()]
    return [
        ("CFR (deaths / cases)", f"{sim_cfr:.4f}", f"{real_cfr:.4f}"),
        ("Death peak date", sim_peak["date"].strftime("%Y-%m-%d"),
         real_peak["date"].strftime("%Y-%m-%d")),
        ("Death peak (daily)", f"{sim_peak['deaths']:.0f}",
         f"{real_peak['deaths']:.0f}"),
    ]

def metrics_table(run):
    rows = compute_metrics(run)
    header = html.Tr([html.Th("Metric"), html.Th(f"Simulation ({run})"),
                      html.Th("Real")])
    body = [html.Tr([html.Td(r[0]), html.Td(r[1]), html.Td(r[2])])
            for r in rows]
    return html.Table([header] + body,
                      style={"width": "100%", "borderCollapse": "collapse",
                             "textAlign": "left"})

def derived_ratios(run):
    s = wave(load_run_daily(run))
    inf = max(s["infections"].sum(), 1)
    hosp = max(s["hospital_admissions"].sum(), 1)
    icu = s["icu_admissions"].sum()
    death = s["deaths"].sum()
    return [
        {"name": "Infection fatality ratio (deaths / infections)",
         "value": f"{death/inf:.4f}  ({death/inf*100:.2f}%)",
         "what": "The proportion of deaths amongst all those infected.",
         "read": "Compare with published COVID IFR estimates for the first "
                 "wave (roughly 0.5-1%)."},
        {"name": "Hospitalisation ratio (admissions / infections)",
         "value": f"{hosp/inf:.4f}  ({hosp/inf*100:.2f}%)",
         "what": "The proportion of all infected individuals requiring hospitalisation.",
         "read": "Checks the first step of the severity pathway."},
        {"name": "ICU ratio (ICU admissions / hospital admissions)",
         "value": f"{icu/hosp:.4f}  ({icu/hosp*100:.2f}%)",
         "what": "The proportion of inpatients admitted to the intensive care unit.",
         "read": "Reported first-wave figures are on the order of 10-17%."},
        {"name": "Hospital fatality ratio (deaths / hospital admissions)",
         "value": f"{death/hosp:.4f}  ({death/hosp*100:.2f}%)",
         "what": "The proportion of deaths amongst inpatients.",
         "read": "Compare with observed hospital mortality for the period."},
    ]

def ratio_card(r):
    return html.Div(
        style={"border": "1px solid #e5e5e5", "borderRadius": "8px",
               "padding": "14px 16px", "marginBottom": "12px",
               "backgroundColor": "#fafafa"},
        children=[
            html.Div(style={"display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "baseline"}, children=[
                html.Span(r["name"], style={"fontWeight": "bold"}),
                html.Span(r["value"], style={"fontSize": "18px",
                                             "color": SIM_COLOR,
                                             "fontWeight": "bold"}),
            ]),
            html.Div("What it is: " + r["what"],
                     style={"fontSize": "13px", "color": "#555",
                            "marginTop": "6px"}),
            html.Div("How to read it: " + r["read"],
                     style={"fontSize": "13px", "color": "#777",
                            "marginTop": "3px"}),
        ])


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
def run_metrics(run):
    sim_w = weekly_deaths(wave(load_run_daily(run)), "date").rename(
        columns={"deaths": "sim"})
    real_w = weekly_deaths(wave(real_daily), "date").rename(
        columns={"deaths": "real"})
    m = pd.merge(sim_w, real_w, on="week", how="inner")
    if len(m) < 3:
        return None
    s, r = m["sim"].values.astype(float), m["real"].values.astype(float)
    return {"run": run,
            "RMSE": round(float(np.sqrt(np.mean((s - r) ** 2))), 1),
            "MAE": round(float(np.mean(np.abs(s - r))), 1),
            "correlation": round(float(np.corrcoef(s, r)[0, 1]), 3)
            if s.std() and r.std() else np.nan,
            "total ratio": round(float(s.sum() / r.sum()), 3)
            if r.sum() else np.nan}

def comparison_figure():
    fig = go.Figure()
    real_w = weekly_deaths(wave(real_daily), "date")
    fig.add_trace(go.Scatter(x=real_w["week"], y=real_w["deaths"],
                             name="Real", mode="lines+markers",
                             line=dict(color=REAL_COLOR, width=3)))
    for run in RUNS:
        sim_w = weekly_deaths(wave(load_run_daily(run)), "date")
        fig.add_trace(go.Scatter(x=sim_w["week"], y=sim_w["deaths"],
                                 name=f"Run {run}", mode="lines"))
    fig.update_layout(
        title="Weekly deaths: all runs vs real data",
        xaxis_title="Week", yaxis_title="Weekly deaths",
        template="simple_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig

def comparison_table():
    rows = [run_metrics(r) for r in RUNS]
    table = pd.DataFrame([r for r in rows if r])
    header = html.Tr([html.Th(c) for c in table.columns])
    body = [html.Tr([html.Td(v) for v in row]) for row in table.values]
    return html.Table([header] + body,
                      style={"width": "100%", "borderCollapse": "collapse",
                             "textAlign": "left"})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    style={"maxWidth": "1000px", "margin": "30px auto",
           "fontFamily": "Arial, sans-serif"},
    children=[
        html.H2("JUNE 2 Simulation vs Real England COVID-19 Data: First Wave"),
        html.Div([
            html.Span("Run: ", style={"fontWeight": "bold"}),
            dcc.Dropdown(id="run-selector",
                         options=[{"label": r, "value": r} for r in RUNS],
                         value=RUNS[0] if RUNS else None, clearable=False,
                         style={"width": "150px", "display": "inline-block"}),
        ], style={"marginBottom": "10px"}),
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview"),
            dcc.Tab(label="Demographics", value="demographics"),
            dcc.Tab(label="Fit", value="fit"),
            dcc.Tab(label="Metrics", value="metrics"),
            dcc.Tab(label="Run comparison", value="comparison"),
            dcc.Tab(label="About", value="about"),
        ]),
        html.Div(id="tab-content", style={"marginTop": "20px"}),
    ])


@app.callback(Output("tab-content", "children"),
              Input("tabs", "value"), Input("run-selector", "value"))
def render_tab(tab, run):
    if tab == "overview":
        return html.Div([
            html.P("Use the run selector to switch settings. "),
            dcc.Dropdown(id="metric-dropdown",
                         options=[{"label": "Cases", "value": "cases"},
                                  {"label": "Deaths", "value": "deaths"}],
                         value="deaths", clearable=False,
                         style={"width": "220px"}),
            dcc.Graph(figure=overview_figure(run, "deaths"), id="overview-chart"),
        ])
    if tab == "demographics":
        return html.Div([
            html.P("'Distribution' compares the shape of deaths across groups. "),
            html.P("'Over time' shows one group's simulated and real deaths "
                   "across the wave, each scaled to its own peak, with policy "
                   "dates."),
            html.P("Real breakdowns are England and Wales level. In summary, "
                   "this is descriptive comparing trends and patterns."),
            html.Div(style={"display": "flex", "gap": "12px"}, children=[
                dcc.Dropdown(id="demo-dropdown",
                             options=[{"label": "By age", "value": "age"},
                                      {"label": "By sex", "value": "sex"}],
                             value="age", clearable=False,
                             style={"width": "180px"}),
                dcc.Dropdown(id="demo-view",
                             options=[{"label": "Distribution", "value": "dist"},
                                      {"label": "Over time", "value": "time"}],
                             value="dist", clearable=False,
                             style={"width": "180px"}),
                html.Div(id="demo-group-wrap", style={"display": "none"},
                         children=[
                    dcc.Dropdown(id="demo-group",
                                 options=[{"label": g, "value": g}
                                          for g in AGE_ORDER],
                                 value="75+", clearable=False,
                                 style={"width": "180px"}),
                ]),
            ]),
            dcc.Graph(figure=demographics_distribution(run, "age"),
                      id="demo-chart"),
        ])
    if tab == "fit":
        return html.Div([
            html.P("Shift the simulation in time to see how well the shapes "
                   "align once any timing offset is removed."),
            html.Div(id="fit-metrics", children=fit_metrics_line(run, 0),
                     style={"fontWeight": "bold", "margin": "10px 0"}),
            dcc.Slider(id="shift-slider", min=0, max=12, step=1, value=0,
                       marks={i: f"{i}w" for i in range(0, 13, 2)}),
            dcc.Graph(figure=fit_figure(run, 0), id="fit-chart"),
        ])
    if tab == "metrics":
        return html.Div([
            html.P("Derived metrics for the selected run versus real data."),
            metrics_table(run),
            html.P("Real cases are undercounted early in the pandemic, so the "
                   "real CFR is likely overstated.",
                   style={"fontSize": "13px", "color": "#666",
                          "margin": "12px 0 25px 0"}),
            html.H4("Severity ratios (simulation internal check)"),
            html.P("These ratios trace the infection -> hospital -> ICU -> "
                   "death pathway inside the simulation. As real-world data "
                   "are not available, the model’s plausibility is "
                   "tverified by comparing its internal severity "
                   "hierarchy with published data.",
                   style={"fontSize": "13px", "color": "#666",
                          "marginBottom": "15px"}),
            html.Div([ratio_card(r) for r in derived_ratios(run)]),
        ])
    if tab == "comparison":
        return html.Div([
            html.P("All runs compared against the real data. The chart shows "
                   "weekly deaths for each run; the table quantifies the "
                   "difference using RMSE, MAE, correlation and the ratio of "
                   "total simulated to total real deaths."),
            dcc.Graph(figure=comparison_figure()),
            html.H4("Quantitative comparison"),
            comparison_table(),
            html.P("Lower RMSE and MAE, correlation closer to 1, and a total "
                   "ratio closer to 1 indicate a better match. No single run "
                   "is best on every measure, so they are read together.",
                   style={"fontSize": "13px", "color": "#666",
                          "marginTop": "12px"}),
        ])
    if tab == "about":
        note = {"marginBottom": "8px"}
        return html.Div([
            html.H4("Data sources", style={
        "fontSize": "20px",
        "fontWeight": "600",
        "marginBottom": "10px",
        "color": SIM_COLOR
    }),
            html.P([html.Strong("Simulation: "),
                   "JUNE 2 event logs for Great Britain under "
                   "different transmissibility multipliers, including m0.4, m0.8 and m1.0, "
                   "first wave from 1 Feb to 1 Aug 2020, provided by the "
                   "supervisor. Each run records individual events, "
                   "including infections, hospital and ICU admissions, deaths in HDF5 "
                   "format, aggregated here into daily and weekly summaries."],
                   style=note),
            html.P([html.Strong("Real cases and deaths: "),
                   "UKHSA dashboard API, England, "
                   "daily data."], style=note),
            html.P([html.Strong("Real deaths by age and sex: "),
                   "ONS weekly deaths by age and "
                   "sex, England and Wales, weekly data."], style=note),
            html.P([html.Strong("Policy dates: "),
                   "A timeline of the UK’s COVID-19 lockdowns and "
                   "crestrictions, published by the Institute for Government."], style=note),
            html.H4("Notes", style={"marginTop": "20px",
                                    "fontSize": "20px",
                                    "fontWeight": "600",
                                    "color": SIM_COLOR}),
            html.P([html.Strong("Time resolution: "),
                   "the Overview shows daily values; the Fit "
                   "and Run comparison tabs aggregate to weekly totals to "
                   "match the weekly real deaths, so those curves are smoother "
                   "and larger in scale."], style=note),
            html.P([html.Strong("Geography: "),
                   "the simulation is Great Britain and the real "
                   "cases and deaths are England, while the age and sex "
                   "breakdowns are England and Wales. Because populations "
                   "differ in size, demographic comparisons use distribution "
                   "shape rather than absolute numbers."], style=note),
            html.P([html.Strong("Cases: "),
                   "real cases are undercounted early in the pandemic "
                   "due to limited testing, so case comparisons are indicative "
                   "of trend rather than level."], style=note),
            html.P([html.Strong("Hospital and ICU admissions "),
                   "are not used for direct "
                   "comparison. National hospital admission data for England "
                   "from NHS England begins only in August 2020 and so does not "
                   "cover the first wave, and early-pandemic figures are reported as "
                   "bed occupancy rather than admissions, on a different basis "
                   "from the simulated admission events. Comparable ICU "
                   "admission data are not available for the period. These "
                   "simulated outputs are therefore used only for the internal "
                   "severity ratios in the Metrics tab."], style=note),
            html.P([html.Strong("Policy timing "),
                   "is shown for context. Any observed associations "
                   "are descriptive and do not in themselves prove the existence "
                   "of a causal relationship."], style=note),
        ])


@app.callback(Output("overview-chart", "figure"),
              Input("run-selector", "value"),
              Input("metric-dropdown", "value"))
def update_overview(run, metric):
    return overview_figure(run, metric)


@app.callback(Output("demo-chart", "figure"),
              Output("demo-group", "options"),
              Output("demo-group", "value"),
              Output("demo-group-wrap", "style"),
              Input("run-selector", "value"),
              Input("demo-dropdown", "value"),
              Input("demo-view", "value"),
              Input("demo-group", "value"))
def update_demo(run, dimension, view, group):
    order = AGE_ORDER if dimension == "age" else SEX_ORDER
    options = [{"label": g, "value": g} for g in order]
    if group not in order:
        group = order[-1]
    if view == "time":
        fig = demographics_over_time(run, dimension, group)
        style = {"display": "block"}
    else:
        fig = demographics_distribution(run, dimension)
        style = {"display": "none"}
    return fig, options, group, style


@app.callback(Output("fit-chart", "figure"),
              Output("fit-metrics", "children"),
              Input("run-selector", "value"),
              Input("shift-slider", "value"))
def update_fit(run, shift):
    return fit_figure(run, shift), fit_metrics_line(run, shift)


if __name__ == "__main__":
    app.run(debug=True)
