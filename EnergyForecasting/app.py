
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecasting")
st.write(
    "A Cloud-Based Energy Consumption Forecasting "
    "and Optimization System Using Machine Learning"
)

st.caption(
    "Analyze historical electricity demand, evaluate "
    "a forecasting model, and estimate future demand."
)


# ==================================================
# 2. DEMO DATA GENERATION
# ==================================================

def create_demo_data():
    """Generate simulated hourly data for testing."""
    rng = np.random.default_rng(42)

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=24 * 90,
        freq="h"
    )

    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    day_number = np.arange(len(timestamps))

    daily_pattern = 40 * np.sin(
        2 * np.pi * hour / 24
    )

    weekly_pattern = np.where(
        weekday < 5, 25, -10
    )

    trend = day_number * 0.015
    noise = rng.normal(0, 8, len(timestamps))

    consumption = (
        200 + daily_pattern + weekly_pattern
        + trend + noise
    )

    consumption = np.maximum(consumption, 10)

    return pd.DataFrame({
        "timestamp": timestamps,
        "energy_consumption": consumption
    })


# ==================================================
# 3. DATA CLEANING
# ==================================================

def clean_data(df):
    """Clean and standardize the uploaded energy dataset."""
    df = df.copy()

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    # Recognize common dataset column names.
    aliases = {
        "datetime": "timestamp",
        "date": "timestamp",
        "time": "timestamp",
        "pjm_load_mw": "energy_consumption",
        "load_mw": "energy_consumption",
        "energy": "energy_consumption",
        "consumption": "energy_consumption",
        "value": "energy_consumption"
    }

    df = df.rename(columns=aliases)

    required = [
        "timestamp",
        "energy_consumption"
    ]

    if not all(column in df.columns for column in required):
        raise ValueError(
            "CSV columns were not recognized. "
            "Expected a timestamp column and an energy "
            "consumption column, such as Datetime and "
            "PJM_Load_MW."
        )

    df = df[required]

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df["energy_consumption"] = pd.to_numeric(
        df["energy_consumption"], errors="coerce"
    )

    df = df.dropna()

    # Average duplicate timestamps.
    df = df.groupby(
        "timestamp", as_index=False
    )["energy_consumption"].mean()

    df = df.sort_values("timestamp")

    # Convert to hourly frequency.
    df = df.set_index("timestamp").resample("h").mean()

    # Fill gaps between observed measurements.
    df["energy_consumption"] = (
        df["energy_consumption"].interpolate(
            method="time"
        )
    )

    df = df.dropna().reset_index()

    if len(df) < 72:
        raise ValueError(
            "At least 72 hourly observations are required."
        )

    if not np.isfinite(
        df["energy_consumption"].to_numpy()
    ).all():
        raise ValueError(
            "The dataset contains invalid energy values."
        )

    return df


# ==================================================
# 4. FEATURE ENGINEERING
# ==================================================

FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "lag_1",
    "lag_24",
    "rolling_mean_24"
]


def create_features(df):
    """Create features from historical observations."""
    result = df.copy()

    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = (
        result["timestamp"].dt.dayofweek
    )
    result["month"] = result["timestamp"].dt.month

    result["lag_1"] = (
        result["energy_consumption"].shift(1)
    )

    result["lag_24"] = (
        result["energy_consumption"].shift(24)
    )

    result["rolling_mean_24"] = (
        result["energy_consumption"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    return result.dropna().reset_index(drop=True)


# ==================================================
# 5. MODEL TRAINING AND EVALUATION
# ==================================================

def train_model(df):
    """Train a Random Forest and evaluate chronologically."""
    featured_data = create_features(df)

    split_index = int(len(featured_data) * 0.8)

    train_data = featured_data.iloc[:split_index]
    test_data = featured_data.iloc[split_index:]

    if len(train_data) < 24 or len(test_data) == 0:
        raise ValueError(
            "Not enough data to train and test the model."
        )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train_data[FEATURES],
        train_data["energy_consumption"]
    )

    predictions = model.predict(test_data[FEATURES])
    actual = test_data["energy_consumption"].to_numpy()

    mae = mean_absolute_error(actual, predictions)

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    nonzero = actual != 0

    mape = None

    if nonzero.any():
        mape = np.mean(
            np.abs(
                (actual[nonzero] - predictions[nonzero])
                / actual[nonzero]
            )
        ) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    comparison = pd.DataFrame({
        "Timestamp": test_data["timestamp"].to_numpy(),
        "Actual": actual,
        "Predicted": predictions
    })

    return model, metrics, comparison


# ==================================================
# 6. NEXT 24-HOUR FORECAST
# ==================================================

def forecast_next_24_hours(model, df):
    """Generate recursive forecasts for the next 24 hours."""
    history = df["energy_consumption"].tolist()
    last_timestamp = df["timestamp"].iloc[-1]

    future_times = []
    future_values = []

    for step in range(1, 25):
        next_time = last_timestamp + pd.Timedelta(
            hours=step
        )

        row = pd.DataFrame([{
            "hour": next_time.hour,
            "day_of_week": next_time.dayofweek,
            "month": next_time.month,
            "lag_1": history[-1],
            "lag_24": history[-24],
            "rolling_mean_24": np.mean(history[-24:])
        }])

        predicted = float(
            model.predict(row[FEATURES])[0]
        )

        predicted = max(0, predicted)

        history.append(predicted)
        future_times.append(next_time)
        future_values.append(predicted)

    return pd.DataFrame({
        "Timestamp": future_times,
        "Forecast": future_values
    })


# ==================================================
# 7. SIDEBAR AND DATA INPUT
# ==================================================

st.sidebar.header("Data Settings")

source = st.sidebar.radio(
    "Choose your data source",
    ["Upload CSV", "Demo dataset"]
)

if source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload electricity data",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info(
            "Choose Upload CSV in the sidebar and upload "
            "your PJM_Load_hourly.csv file."
        )
        st.stop()

    try:
        raw_data = pd.read_csv(uploaded_file)
        data = clean_data(raw_data)

    except Exception as error:
        st.error(f"Could not load the dataset: {error}")
        st.stop()

    st.success("Uploaded dataset loaded successfully.")

else:
    data = create_demo_data()

    st.warning(
        "Demo mode: these are simulated values, "
        "not actual electricity measurements."
    )


# ==================================================
# 8. PROJECT ANALYSIS
# ==================================================

st.header("1. Project Analysis")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Hourly records",
    f"{len(data):,}"
)

col2.metric(
    "Average consumption",
    f"{data['energy_consumption'].mean():,.2f}"
)

col3.metric(
    "Maximum consumption",
    f"{data['energy_consumption'].max():,.2f}"
)

st.subheader("Historical Energy Consumption")

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    data["timestamp"],
    data["energy_consumption"],
    linewidth=1
)

ax.set_xlabel("Timestamp")
ax.set_ylabel("Energy consumption (dataset units)")
ax.grid(True, alpha=0.3)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

with st.expander("View cleaned dataset"):
    st.dataframe(data, use_container_width=True)


# ==================================================
# 9. MACHINE LEARNING
# ==================================================

st.header("2. Machine Learning Implementation")

try:
    with st.spinner("Training Random Forest model..."):
        model, metrics, comparison = train_model(data)

        forecast = forecast_next_24_hours(model, data)

except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()

st.success("Model training and evaluation completed.")

st.caption(
    "The test period follows the training period "
    "chronologically. Evaluation uses observed lag values "
    "for each test timestamp."
)

m1, m2, m3 = st.columns(3)

m1.metric("MAE", f"{metrics['MAE']:.3f}")
m2.metric("RMSE", f"{metrics['RMSE']:.3f}")

if metrics["MAPE"] is not None:
    m3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
else:
    m3.metric("MAPE", "Unavailable")


# ==================================================
# 10. ACTUAL VS PREDICTED
# ==================================================

st.subheader("Actual vs Predicted Consumption")

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    comparison["Timestamp"],
    comparison["Actual"],
    label="Actual"
)

ax.plot(
    comparison["Timestamp"],
    comparison["Predicted"],
    label="Predicted"
)

ax.set_xlabel("Timestamp")
ax.set_ylabel("Energy consumption")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

with st.expander("View evaluation predictions"):
    st.dataframe(comparison, use_container_width=True)


# ==================================================
# 11. FUTURE FORECAST
# ==================================================

st.header("3. Next 24-Hour Energy Forecast")

st.write(
    "The model predicts each future hour using historical "
    "values and its earlier predictions."
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    forecast["Timestamp"],
    forecast["Forecast"],
    marker="o"
)

ax.set_xlabel("Future timestamp")
ax.set_ylabel("Forecast consumption")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, alpha=0.3)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.dataframe(forecast, use_container_width=True)

st.download_button(
    label="Download 24-hour forecast CSV",
    data=forecast.to_csv(index=False).encode("utf-8"),
    file_name="energy_forecast.csv",
    mime="text/csv"
)


# ==================================================
# 12. BASIC OPTIMIZATION RECOMMENDATIONS
# ==================================================

st.header("4. Energy Optimization Recommendations")

average_forecast = forecast["Forecast"].mean()
peak_forecast = forecast["Forecast"].max()

c1, c2 = st.columns(2)

c1.metric(
    "Average forecast consumption",
    f"{average_forecast:.2f}"
)

c2.metric(
    "Peak forecast consumption",
    f"{peak_forecast:.2f}"
)

if peak_forecast > average_forecast * 1.2:
    st.warning(
        "A high-demand hour is forecast. Consider shifting "
        "flexible electricity use to lower-demand hours."
    )
else:
    st.info(
        "The forecast did not show a peak more than 20% "
        "above its average using this simple rule."
    )

peak_hours = forecast[
    forecast["Forecast"] > average_forecast * 1.2
]

if not peak_hours.empty:
    st.subheader("Hours to Review")

    st.dataframe(
        peak_hours,
        use_container_width=True
    )

st.caption(
    "These are rule-based recommendations. The system "
    "does not automatically control electrical equipment."
)

st.divider()

st.caption(
    "Prototype: local Streamlit application. "
    "Cloud deployment and multi-cloud redundancy "
    "require separate implementation and testing."
)