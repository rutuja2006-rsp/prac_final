

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="CloudBudget Guard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- DESIGN --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: linear-gradient(135deg,#f5f7ff 0%,#f9fbff 55%,#f1f7ff 100%);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
h1, h2, h3 {
    font-family: 'Manrope', sans-serif;
    color: #172554;
}
.hero {
    background: linear-gradient(120deg,#172554,#4338ca,#0e7490);
    padding: 28px 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(49,46,129,.15);
}
.hero h1 {color:white; font-size:34px; margin-bottom:5px;}
.hero p {color:#e0e7ff; margin-bottom:0;}
.metric-card {
    background:white;
    border:1px solid #e3e8f7;
    border-radius:18px;
    padding:20px;
    box-shadow:0 5px 18px rgba(30,41,59,.04);
    min-height:120px;
}
.metric-label {font-size:13px;color:#64748b;font-weight:600;}
.metric-value {font-size:27px;color:#172554;font-weight:800;margin-top:8px;}
.metric-note {font-size:12px;color:#64748b;margin-top:4px;}
.section-card {
    background:white;
    border:1px solid #e2e8f0;
    border-radius:18px;
    padding:20px;
    margin-bottom:16px;
}
div[data-testid="stMetric"] {
    background:white;
    border:1px solid #e2e8f0;
    border-radius:16px;
    padding:16px;
}
div[data-testid="stMetricValue"] {color:#312e81;}
.stButton button {
    border-radius:10px;
    font-weight:600;
}
[data-testid="stSidebar"] {
    background:#eef2ff;
}
.small-muted {color:#64748b;font-size:13px;}
</style>
""", unsafe_allow_html=True)


def money(value, currency):
    try:
        value = float(value)
    except (ValueError, TypeError):
        value = 0.0
    return f"{currency} {value:,.2f}"


def metric_card(label, value, note=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>""",
        unsafe_allow_html=True
    )


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]

    aliases = {
        "date": "charge_date",
        "billing_date": "charge_date",
        "usage_date": "charge_date",
        "invoice_date": "charge_date",
        "amount": "cost",
        "spend": "cost",
        "total_cost": "cost",
        "service": "service_name",
        "provider_name": "provider",
        "cloud_provider": "provider",
        "currency_code": "currency",
        "cpu_utilization": "cpu_utilization_pct",
        "cpu_usage": "cpu_utilization_pct",
        "memory_utilization": "memory_utilization_pct",
        "memory_usage": "memory_utilization_pct",
        "resource": "resource_id",
        "team": "team_id"
    }

    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df


def prepare_data(df):
    df = normalize_columns(df)

    if "charge_date" not in df.columns or "cost" not in df.columns:
        return None, (
            "CSV must contain a date column and a cost column. "
            "Accepted examples: charge_date/date/billing_date and cost/amount."
        )

    df["charge_date"] = pd.to_datetime(df["charge_date"], errors="coerce")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df = df.dropna(subset=["charge_date", "cost"]).copy()

    if df.empty:
        return None, "No valid rows found after parsing dates and costs."

    defaults = {
        "provider": "Unknown",
        "service_name": "Unknown",
        "currency": "USD",
        "team_id": "Unknown",
        "resource_id": "Unknown"
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)

    for col in [
        "budget_amount",
        "cpu_utilization_pct",
        "memory_utilization_pct",
        "usage_quantity",
        "network_in_gb",
        "network_out_gb"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["month"] = df["charge_date"].dt.to_period("M").astype(str)
    df["year"] = df["charge_date"].dt.year
    df["month_number"] = df["charge_date"].dt.month
    return df, None


def aggregate_monthly(df):
    return (
        df.groupby("month", as_index=False)
        .agg(cost=("cost", "sum"))
        .sort_values("month")
    )


def detect_anomalies(df, contamination=0.05):
    out = df.copy()

    # Isolation Forest compares each record with the overall distribution
    # of cost values; it does not require a manually supplied expected cost.
    if len(out) < 5 or out["cost"].nunique() < 2:
        out["anomaly_flag"] = 0
        out["anomaly_score"] = 0.0
    else:
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )
        features = out[["cost"]].fillna(0)
        prediction = model.fit_predict(features)
        out["anomaly_flag"] = (prediction == -1).astype(int)
        out["anomaly_score"] = -model.decision_function(features)

    median_cost = out["cost"].median()
    q1 = out["cost"].quantile(0.25)
    q3 = out["cost"].quantile(0.75)
    iqr = q3 - q1
    upper_outlier = q3 + 1.5 * iqr

    reasons = []
    for _, row in out.iterrows():
        reason = []

        if row["anomaly_flag"] == 1:
            reason.append("Isolation Forest identified an unusual cost pattern")

        if row["cost"] > upper_outlier and iqr > 0:
            reason.append("Cost is unusually high compared with the cost distribution")

        if row["cost"] > median_cost * 2 and median_cost > 0:
            reason.append("Cost is more than twice the median cost")

        cpu = row.get("cpu_utilization_pct", np.nan)
        if pd.notna(cpu) and cpu < 20 and row["cost"] > median_cost:
            reason.append("High cost with low CPU utilization; review resource sizing")

        if not reason:
            reason.append("No strong anomaly reason detected")

        reasons.append("; ".join(reason))

    out["reason_for_flagging"] = reasons

    if "review_status" not in out.columns:
        out["review_status"] = np.where(
            out["anomaly_flag"] == 1,
            "Needs investigation",
            "Not flagged"
        )

    return out


def forecast_costs(monthly, periods=12):
    monthly = monthly.copy().sort_values("month")
    monthly["month_date"] = pd.to_datetime(monthly["month"] + "-01")
    monthly = monthly.groupby("month_date", as_index=False)["cost"].sum()
    monthly = monthly.sort_values("month_date")

    if len(monthly) < 6:
        return None, None, (
            "At least 6 months of history are needed to train and compare "
            "forecasting models."
        )

    monthly["time_index"] = np.arange(len(monthly))
    monthly["month_number"] = monthly["month_date"].dt.month

    X = monthly[["time_index", "month_number"]]
    y = monthly["cost"].values

    split = max(3, int(len(monthly) * 0.8))
    split = min(split, len(monthly) - 1)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=1
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150,
            random_state=42
        )
    }

    metrics = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = np.maximum(model.predict(X_test), 0)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        denominator = np.sum(np.abs(y_test))
        wape = (
            np.sum(np.abs(y_test - predictions)) / denominator * 100
            if denominator else np.nan
        )

        metrics.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "WAPE (%)": wape
        })
        trained_models[name] = model

    metrics_df = pd.DataFrame(metrics).sort_values("WAPE (%)")
    best_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    # Refit the selected model on all available history.
    best_model.fit(X, y)

    future_dates = pd.date_range(
        monthly["month_date"].max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS"
    )
    future = pd.DataFrame({
        "time_index": np.arange(len(monthly), len(monthly) + periods),
        "month_number": future_dates.month
    })
    future["predicted_cost"] = np.maximum(best_model.predict(future), 0)
    future["month"] = future_dates.strftime("%Y-%m")

    return future, metrics_df, None


def download_csv(df, filename):
    st.download_button(
        label=f"⬇️ Download {filename}",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## ☁️ CloudBudget Guard")
    st.caption("Cloud spending intelligence")
    uploaded_file = st.file_uploader("Upload cloud cost CSV", type=["csv"])

    st.markdown("---")
    st.markdown("### What this app does")
    st.caption("📊 Monthly and yearly spend tracking")
    st.caption("🔮 12-month cost forecasting")
    st.caption("🚨 Isolation Forest anomaly detection")
    st.caption("🎯 Budget monitoring")
    st.caption("💡 Cost optimization investigation")


if uploaded_file is None:
    st.markdown("""
    <div class="hero">
        <h1>☁️ CloudBudget Guard</h1>
        <p>Forecast cloud spending, investigate unusual costs, and monitor budgets.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Upload your cloud billing CSV using the sidebar to open the dashboard.")

    st.markdown("### Expected data format")
    sample = pd.DataFrame({
        "charge_date": ["2026-01-01", "2026-01-02", "2026-02-01"],
        "cost": [125.50, 142.00, 168.20],
        "provider": ["AWS", "AWS", "Azure"],
        "service_name": ["Compute", "Storage", "Compute"],
        "currency": ["USD", "USD", "USD"],
        "team_id": ["Team A", "Team A", "Team B"],
        "resource_id": ["vm-001", "disk-002", "vm-003"],
        "cpu_utilization_pct": [15, 60, 12],
        "budget_amount": [1000, 1000, 1200]
    })
    st.dataframe(sample, use_container_width=True, hide_index=True)
    st.stop()


try:
    raw_df = pd.read_csv(uploaded_file)
    df, error = prepare_data(raw_df)
except Exception as e:
    st.error(f"Could not read this CSV: {e}")
    st.stop()

if error:
    st.error(error)
    st.stop()


# -------------------- FILTERS --------------------

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔎 Filters")

    currencies = sorted(df["currency"].unique().tolist())
    selected_currency = st.selectbox("Currency", currencies)

    providers = sorted(df["provider"].unique().tolist())
    selected_providers = st.multiselect(
        "Cloud provider",
        providers,
        default=providers
    )

    services = sorted(df["service_name"].unique().tolist())
    selected_services = st.multiselect(
        "Service",
        services,
        default=services
    )

    min_date = df["charge_date"].min().date()
    max_date = df["charge_date"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    contamination = st.slider(
        "Anomaly sensitivity",
        min_value=0.01,
        max_value=0.15,
        value=0.05,
        step=0.01,
        help="Higher values flag a larger proportion of records as unusual."
    )

filtered = df[df["currency"] == selected_currency].copy()
filtered = filtered[filtered["provider"].isin(selected_providers)]
filtered = filtered[filtered["service_name"].isin(selected_services)]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[
        (filtered["charge_date"].dt.date >= date_range[0]) &
        (filtered["charge_date"].dt.date <= date_range[1])
    ]

if filtered.empty:
    st.warning("No records match the selected filters. Change the filters in the sidebar.")
    st.stop()

filtered["month"] = filtered["charge_date"].dt.to_period("M").astype(str)
filtered["year"] = filtered["charge_date"].dt.year


# -------------------- NAVIGATION --------------------

st.markdown("""
<div class="hero">
    <h1>☁️ CloudBudget Guard</h1>
    <p>Understand spending patterns. Predict future costs. Investigate unusual activity.</p>
</div>
""", unsafe_allow_html=True)

pages = [
    "Overview",
    "Monthly & Yearly Tracking",
    "Cost Forecasting",
    "Anomaly Detection",
    "Budget Tracking",
    "Recommendations",
    "Data Explorer"
]

page = st.radio(
    "Navigate dashboard",
    pages,
    horizontal=True,
    label_visibility="collapsed"
)

st.caption(
    f"Showing {len(filtered):,} billing records · "
    f"{selected_currency} · "
    f"{filtered['charge_date'].min().date()} to {filtered['charge_date'].max().date()}"
)

monthly = aggregate_monthly(filtered)
total_cost = filtered["cost"].sum()
daily_avg = filtered.groupby(filtered["charge_date"].dt.date)["cost"].sum().mean()
unique_services = filtered["service_name"].nunique()


# -------------------- OVERVIEW --------------------

if page == "Overview":
    st.subheader("📊 Spending Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Spend", money(total_cost, selected_currency), "For selected filters")
    with col2:
        metric_card("Average Daily Spend", money(daily_avg, selected_currency), "Average across active dates")
    with col3:
        metric_card("Services", f"{unique_services}", "Unique services in selection")
    with col4:
        metric_card("Billing Records", f"{len(filtered):,}", "Rows included in analysis")

    st.markdown("### Spending trend")
    if not monthly.empty:
        fig = px.area(
            monthly, x="month", y="cost",
            markers=True,
            title="Monthly cloud spend",
            labels={"month": "Month", "cost": f"Cost ({selected_currency})"}
        )
        fig.update_traces(line_color="#4f46e5", fillcolor="rgba(99,102,241,0.16)")
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Spend by provider")
        provider_cost = (
            filtered.groupby("provider", as_index=False)["cost"]
            .sum().sort_values("cost", ascending=False)
        )
        fig = px.pie(
            provider_cost,
            names="provider",
            values="cost",
            hole=0.55,
            title="Provider cost distribution"
        )
        fig.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Top services by spend")
        service_cost = (
            filtered.groupby("service_name", as_index=False)["cost"]
            .sum().sort_values("cost", ascending=True).tail(10)
        )
        fig = px.bar(
            service_cost, x="cost", y="service_name",
            orientation="h",
            color="cost",
            color_continuous_scale="Purples",
            title="Highest-cost services",
            labels={"cost": f"Cost ({selected_currency})", "service_name": "Service"}
        )
        fig.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recent billing records")
    st.dataframe(
        filtered.sort_values("charge_date", ascending=False).head(15),
        use_container_width=True,
        hide_index=True
    )


# -------------------- MONTHLY & YEARLY TRACKING --------------------

elif page == "Monthly & Yearly Tracking":
    st.subheader("📅 Monthly & Yearly Cost Tracking")

    monthly_full = (
        filtered.groupby(["year", "month"], as_index=False)
        .agg(
            total_cost=("cost", "sum"),
            record_count=("cost", "size"),
            average_cost=("cost", "mean")
        )
        .sort_values("month")
    )

    st.markdown("#### Monthly history")
    st.dataframe(monthly_full, use_container_width=True, hide_index=True)
    download_csv(monthly_full, "monthly_cost_history.csv")

    fig = px.bar(
        monthly_full, x="month", y="total_cost",
        color="year",
        barmode="group",
        title="Monthly spending history",
        labels={"month": "Month", "total_cost": f"Cost ({selected_currency})"}
    )
    fig.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    yearly = (
        filtered.groupby("year", as_index=False)
        .agg(
            annual_cost=("cost", "sum"),
            record_count=("cost", "size"),
            average_record_cost=("cost", "mean")
        )
        .sort_values("year")
    )

    yearly["year_over_year_change_pct"] = (
        yearly["annual_cost"].pct_change() * 100
    )

    st.markdown("#### Yearly summary")
    st.dataframe(yearly, use_container_width=True, hide_index=True)
    download_csv(yearly, "yearly_cost_summary.csv")

    if len(yearly) >= 2:
        fig = px.bar(
            yearly, x="year", y="annual_cost",
            color="annual_cost",
            color_continuous_scale="Teal",
            title="Annual spending comparison",
            labels={"annual_cost": f"Cost ({selected_currency})"}
        )
        fig.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload data covering multiple years to compare year-over-year spending.")

    st.markdown("#### Breakdown by provider and service")
    breakdown = (
        filtered.groupby(["year", "provider", "service_name"], as_index=False)
        .agg(total_cost=("cost", "sum"))
        .sort_values("total_cost", ascending=False)
    )
    st.dataframe(breakdown, use_container_width=True, hide_index=True)
    download_csv(breakdown, "provider_service_cost_breakdown.csv")


# -------------------- FORECASTING --------------------

elif page == "Cost Forecasting":
    st.subheader("🔮 Cloud Cost Forecasting")
    st.write(
        "Forecasts are estimates based on historical monthly costs. "
        "They are not guaranteed future bills."
    )

    history = aggregate_monthly(filtered)
    future, model_metrics, forecast_error = forecast_costs(history, periods=12)

    if forecast_error:
        st.warning(forecast_error)
        st.info("Add more months of historical billing data to enable model comparison.")
    else:
        best_model = model_metrics.iloc[0]["Model"]
        annual_forecast = future["predicted_cost"].sum()

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("12-Month Forecast", money(annual_forecast, selected_currency),
                        "Sum of next 12 monthly predictions")
        with c2:
            metric_card("Selected Model", best_model, "Lowest holdout WAPE")
        with c3:
            metric_card("Forecast Horizon", "12 months", "From the next month")

        chart_history = history.copy()
        chart_history["Type"] = "Historical"
        chart_history = chart_history.rename(columns={"cost": "amount"})

        chart_future = future[["month", "predicted_cost"]].copy()
        chart_future["Type"] = "Forecast"
        chart_future = chart_future.rename(columns={"predicted_cost": "amount"})

        chart_data = pd.concat(
            [chart_history[["month", "amount", "Type"]],
             chart_future[["month", "amount", "Type"]]],
            ignore_index=True
        )

        fig = px.line(
            chart_data,
            x="month", y="amount",
            color="Type",
            markers=True,
            title="Historical spending and 12-month forecast",
            labels={"month": "Month", "amount": f"Cost ({selected_currency})"}
        )
        fig.update_layout(paper_bgcolor="white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Monthly predictions")
        st.dataframe(
            future.rename(columns={"predicted_cost": "predicted_cost"}),
            use_container_width=True,
            hide_index=True
        )
        download_csv(future, "12_month_cost_forecast.csv")

        yearly_forecast = future.copy()
        yearly_forecast["year"] = pd.to_datetime(
            yearly_forecast["month"] + "-01"
        ).dt.year
        yearly_forecast = (
            yearly_forecast.groupby("year", as_index=False)
            .agg(predicted_annual_cost=("predicted_cost", "sum"))
        )

        st.markdown("#### Forecast by calendar year")
        st.dataframe(yearly_forecast, use_container_width=True, hide_index=True)
        download_csv(yearly_forecast, "annual_cost_forecast.csv")

        st.markdown("#### Model comparison")
        st.caption("Metrics are calculated on a chronological holdout period.")
        st.dataframe(model_metrics, use_container_width=True, hide_index=True)
        download_csv(model_metrics, "forecast_model_comparison.csv")

        st.warning(
            "Only month index and month number are used as model features. "
            "With limited history, forecasts may not capture new contracts, "
            "pricing changes, usage growth, or seasonality."
        )


# -------------------- ANOMALY DETECTION --------------------

elif page == "Anomaly Detection":
    st.subheader("🚨 Anomaly Detection & Investigation")

    st.write(
        "Isolation Forest identifies records that look unusual relative to the "
        "cost distribution. A flagged record is a review candidate, not proof "
        "of fraud or an incorrect bill."
    )

    anomalies = detect_anomalies(filtered, contamination=contamination)

    flagged = anomalies[anomalies["anomaly_flag"] == 1].copy()
    flagged_count = len(flagged)
    flagged_cost = flagged["cost"].sum()
    pct_flagged = flagged_count / len(anomalies) * 100 if len(anomalies) else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Flagged Records", f"{flagged_count:,}", "Records identified as unusual")
    with c2:
        metric_card("Flagged Cost", money(flagged_cost, selected_currency),
                    "Sum of flagged record costs")
    with c3:
        metric_card("Flagged Share", f"{pct_flagged:.1f}%", "Share of records flagged")

    scatter = px.scatter(
        anomalies,
        x="charge_date",
        y="cost",
        color=anomalies["anomaly_flag"].map({0: "Normal", 1: "Flagged"}),
        hover_data=["provider", "service_name", "resource_id", "anomaly_score"],
        title="Cost records over time",
        labels={"cost": f"Cost ({selected_currency})", "charge_date": "Date",
                "color": "Detection result"},
        color_discrete_map={"Normal": "#818cf8", "Flagged": "#ef4444"}
    )
    scatter.update_layout(paper_bgcolor="white")
    st.plotly_chart(scatter, use_container_width=True)

    st.markdown("#### Flagged records and reasons")
    if flagged.empty:
        st.success("No records were flagged with the current anomaly sensitivity.")
    else:
        display_cols = [
            c for c in [
                "charge_date", "provider", "service_name", "team_id",
                "resource_id", "cost", "cpu_utilization_pct",
                "memory_utilization_pct", "anomaly_score",
                "reason_for_flagging", "review_status"
            ] if c in flagged.columns
        ]
        st.dataframe(
            flagged[display_cols].sort_values("anomaly_score", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("#### Review a flagged record")
    if not flagged.empty:
        flagged = flagged.reset_index(drop=True)
        record_labels = [
            f"{i} | {row['charge_date'].date()} | {row['service_name']} | "
            f"{money(row['cost'], selected_currency)}"
            for i, row in flagged.iterrows()
        ]
        selected_record = st.selectbox("Select record", record_labels)
        idx = record_labels.index(selected_record)
        selected_row = flagged.iloc[idx]

        st.info(f"**Why flagged:** {selected_row['reason_for_flagging']}")

        review_options = [
            "Needs investigation",
            "Legitimate cost",
            "Confirmed unusual"
        ]
        current_status = selected_row.get("review_status", "Needs investigation")
        default_idx = review_options.index(current_status) if current_status in review_options else 0

        new_status = st.selectbox(
            "Reviewer status",
            review_options,
            index=default_idx
        )

        if st.button("Save review status"):
            if "review_status_map" not in st.session_state:
                st.session_state["review_status_map"] = {}
            key = (
                str(selected_row["charge_date"]) +
                str(selected_row["service_name"]) +
                str(selected_row["cost"])
            )
            st.session_state["review_status_map"][key] = new_status
            st.success("Review status saved for this session.")

    st.markdown("#### Export anomaly results")
    download_csv(anomalies, "anomaly_detection_results.csv")

    st.caption(
        "Review status is stored in the current Streamlit session. "
        "It is not written back to your original CSV or a persistent database."
    )


# -------------------- BUDGET TRACKING --------------------

elif page == "Budget Tracking":
    st.subheader("🎯 Budget Tracking & Alerts")

    monthly_spend = aggregate_monthly(filtered)
    budget_col, threshold_col = st.columns(2)

    with budget_col:
        default_budget = 1000.0
        if "budget_amount" in filtered.columns:
            budget_values = filtered["budget_amount"].dropna()
            if not budget_values.empty:
                default_budget = float(budget_values.iloc[-1])

        monthly_budget = st.number_input(
            f"Monthly budget ({selected_currency})",
            min_value=0.0,
            value=max(0.0, default_budget),
            step=100.0
        )

    with threshold_col:
        alert_threshold = st.slider(
            "Early warning threshold (%)",
            min_value=50,
            max_value=100,
            value=80,
            step=5
        )

    if monthly_spend.empty:
        st.info("No monthly spend data available.")
    else:
        latest = monthly_spend.iloc[-1]
        latest_spend = float(latest["cost"])
        remaining = monthly_budget - latest_spend
        utilization = latest_spend / monthly_budget * 100 if monthly_budget > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Latest Month", str(latest["month"]), "Most recent month in filtered data")
        with c2:
            metric_card("Actual Spend", money(latest_spend, selected_currency), "Latest month")
        with c3:
            metric_card("Monthly Budget", money(monthly_budget, selected_currency), "Budget entered above")
        with c4:
            metric_card("Budget Used", f"{utilization:.1f}%", "Latest month")

        if monthly_budget > 0 and latest_spend > monthly_budget:
            st.error(
                f"🚨 Over budget by {money(latest_spend - monthly_budget, selected_currency)}."
            )
        elif monthly_budget > 0 and utilization >= alert_threshold:
            st.warning(
                f"⚠️ Spending has reached {utilization:.1f}% of the budget."
            )
        elif monthly_budget > 0:
            st.success(
                f"Within budget. Remaining: {money(remaining, selected_currency)}."
            )
        else:
            st.warning("Set a budget above zero to calculate budget utilization.")

        monthly_spend["budget"] = monthly_budget
        monthly_spend["remaining_budget"] = monthly_budget - monthly_spend["cost"]
        monthly_spend["budget_utilization_pct"] = np.where(
            monthly_budget > 0,
            monthly_spend["cost"] / monthly_budget * 100,
            np.nan
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_spend["month"],
            y=monthly_spend["cost"],
            name="Actual spend",
            marker_color="#6366f1"
        ))
        fig.add_trace(go.Scatter(
            x=monthly_spend["month"],
            y=monthly_spend["budget"],
            name="Monthly budget",
            mode="lines",
            line=dict(color="#f97316", width=3, dash="dash")
        ))
        fig.update_layout(
            title="Actual monthly spend vs budget",
            xaxis_title="Month",
            yaxis_title=f"Cost ({selected_currency})",
            paper_bgcolor="white",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Budget history")
        st.dataframe(monthly_spend, use_container_width=True, hide_index=True)
        download_csv(monthly_spend, "budget_tracking_history.csv")

    st.caption(
        "The entered budget is applied to each month for comparison. "
        "It is not automatically inferred from billing data."
    )


# -------------------- RECOMMENDATIONS --------------------

elif page == "Recommendations":
    st.subheader("💡 Cost Optimization Investigation")

    st.write(
        "These suggestions identify areas worth investigating. "
        "They do not guarantee savings; validate business requirements before changes."
    )

    service_spend = (
        filtered.groupby("service_name", as_index=False)
        .agg(
            total_cost=("cost", "sum"),
            average_cost=("cost", "mean"),
            records=("cost", "size")
        )
        .sort_values("total_cost", ascending=False)
    )

    st.markdown("#### Highest-cost services")
    st.dataframe(service_spend.head(10), use_container_width=True, hide_index=True)

    if not service_spend.empty:
        top_service = service_spend.iloc[0]
        st.info(
            f"**Investigate:** {top_service['service_name']} has the highest "
            f"total spend in the current selection "
            f"({money(top_service['total_cost'], selected_currency)}). "
            "Review usage patterns, pricing tiers, idle resources, and reserved-use options."
        )

    if "cpu_utilization_pct" in filtered.columns:
        cpu_df = filtered.dropna(subset=["cpu_utilization_pct"]).copy()
        low_cpu = cpu_df[
            (cpu_df["cpu_utilization_pct"] < 20) &
            (cpu_df["cost"] > 0)
        ]

        st.markdown("#### Potential high-cost, low-CPU records")
        if low_cpu.empty:
            st.info(
                "No records match the current low-CPU rule. "
                "CPU data may be missing or no resources fall below the threshold."
            )
        else:
            st.write(
                f"{len(low_cpu)} records have CPU utilization below 20% "
                "and positive cost."
            )
            low_cpu = low_cpu.sort_values("cost", ascending=False)
            st.dataframe(
                low_cpu[
                    [c for c in [
                        "charge_date", "provider", "service_name", "resource_id",
                        "cost", "cpu_utilization_pct", "memory_utilization_pct"
                    ] if c in low_cpu.columns]
                ].head(50),
                use_container_width=True,
                hide_index=True
            )
            st.warning(
                "Low CPU alone does not prove that a resource is unnecessary. "
                "Check peak usage, latency, memory, availability requirements, "
                "and workload schedules before rightsizing or shutting it down."
            )
    else:
        st.info(
            "CPU utilization data is not present. Add a cpu_utilization_pct "
            "column to investigate potentially underutilized resources."
        )

    st.markdown("#### Suggested investigation checklist")
    st.markdown("""
    - Review the most expensive services and compare them with previous months.
    - Check for idle or oversized compute resources.
    - Investigate unexpected increases in storage, data transfer, or requests.
    - Review commitment discounts and pricing tiers with the cloud provider.
    - Check tags and team ownership for resources with unclear responsibility.
    - Confirm any proposed change with the service owner before implementation.
    """)

    download_csv(service_spend, "service_optimization_review.csv")


# -------------------- DATA EXPLORER --------------------

elif page == "Data Explorer":
    st.subheader("🧾 Data Explorer")

    st.markdown("#### Filtered data")
    st.dataframe(
        filtered.sort_values("charge_date", ascending=False),
        use_container_width=True,
        hide_index=True
    )
    download_csv(filtered, "filtered_cloud_cost_data.csv")

    st.markdown("#### Dataset information")
    info_df = pd.DataFrame({
        "Column": filtered.columns,
        "Data type": [str(filtered[c].dtype) for c in filtered.columns],
        "Missing values": [int(filtered[c].isna().sum()) for c in filtered.columns],
        "Unique values": [int(filtered[c].nunique()) for c in filtered.columns]
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.markdown("#### Descriptive statistics")
    st.dataframe(
        filtered.describe(include="all").transpose().fillna(""),
        use_container_width=True
    )

    st.markdown("#### Data quality")
    quality = pd.DataFrame({
        "Check": [
            "Rows loaded",
            "Duplicate rows",
            "Missing charge dates",
            "Missing costs",
            "Negative cost records",
            "Distinct currencies"
        ],
        "Result": [
            len(filtered),
            int(filtered.duplicated().sum()),
            int(filtered["charge_date"].isna().sum()),
            int(filtered["cost"].isna().sum()),
            int((filtered["cost"] < 0).sum()),
            int(filtered["currency"].nunique())
        ]
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "CloudBudget Guard · Estimates and anomaly flags are decision-support signals, "
    "not guarantees. All totals are calculated for the selected currency and filters."
)
