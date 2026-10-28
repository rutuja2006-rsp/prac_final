# prac_final
%%writefile /content/README.md
# ☁️ CloudBudget Guard

CloudBudget Guard is a cloud cost monitoring and forecasting dashboard built using Python and Streamlit.

## Features

- Monthly and yearly cloud spending analysis
- 12-month cloud cost forecasting
- Random Forest and HistGradientBoosting model comparison
- Isolation Forest-based anomaly detection
- Anomaly explanations and review statuses
- Monthly budget tracking and alerts
- Cost optimization recommendations
- Interactive charts and CSV exports

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly

## Run Locally

1. Clone the repository.
2. Install dependencies:

   pip install -r requirements.txt

3. Run the application:

   streamlit run app.py

## Dataset

Upload a cloud billing CSV through the sidebar. The dataset should contain date and cost columns. Optional columns can include provider, service, currency, CPU utilization, and budget.

## Disclaimer

Forecasts and anomaly flags are estimates intended to support investigation. They do not guarantee future costs or prove billing errors.
