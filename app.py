from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "ensemble_models.joblib"


MODEL_METRICS = [
    {"model": "Decision Tree", "mae": 21.192604, "mse": 2586.467901, "rmse": 50.857329, "r2": 0.847558},
    {"model": "Bagging Regressor", "mae": 19.045364, "mse": 1860.576011, "rmse": 43.134395, "r2": 0.890341},
    {"model": "Random Forest", "mae": 19.490006, "mse": 1874.094045, "rmse": 43.290808, "r2": 0.889544},
    {"model": "Gradient Boosting", "mae": 19.479374, "mse": 1882.841764, "rmse": 43.391725, "r2": 0.889029},
    {"model": "XGBoost", "mae": 21.126665, "mse": 2663.656250, "rmse": 51.610621, "r2": 0.843009},
    {"model": "Stacking Regressor", "mae": 19.371803, "mse": 1820.712782, "rmse": 42.669811, "r2": 0.892690},
]

FEATURE_IMPORTANCE = [
    {"feature": "Current consumption", "importance": 43.3824},
    {"feature": "Previous month", "importance": 23.7168},
    {"feature": "Two months ago", "importance": 18.3374},
    {"feature": "Three months ago", "importance": 13.4558},
    {"feature": "Month number", "importance": 0.8666},
    {"feature": "Year", "importance": 0.2410},
]


@st.cache_resource
def load_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model file not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    for model in bundle.values():
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    return bundle


def predict(bundle, values):
    input_data = pd.DataFrame(
        [[
            values["consumption"],
            values["consumption_lag_1"],
            values["consumption_lag_2"],
            values["consumption_lag_3"],
            values["month_number"],
            values["year"],
        ]],
        columns=bundle["feature_columns"],
    )

    model_mapping = {
        "Decision Tree": "decision_tree",
        "Bagging Regressor": "bagging",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "XGBoost": "xgboost",
    }

    predictions = {
        name: round(float(bundle[key].predict(input_data)[0]), 2)
        for name, key in model_mapping.items()
    }

    stacking_features = pd.DataFrame({
        "Bagging": [predictions["Bagging Regressor"]],
        "Random Forest": [predictions["Random Forest"]],
        "Gradient Boosting": [predictions["Gradient Boosting"]],
    })
    predictions["Stacking Regressor"] = round(
        float(bundle["stacking_meta"].predict(stacking_features)[0]),
        2,
    )
    return predictions


st.set_page_config(
    page_title="EnsembleLab",
    page_icon="zap",
    layout="wide",
)

st.title("EnsembleLab")
st.caption("Sri Lankan Household Electricity Consumption")

try:
    model_bundle = load_models()
except Exception as error:
    st.error(str(error))
    st.stop()

page = st.sidebar.radio(
    "Workspace",
    ["Overview", "Model Comparison", "Feature Analysis", "Prediction"],
)

metrics_df = pd.DataFrame(MODEL_METRICS)
feature_df = pd.DataFrame(FEATURE_IMPORTANCE)

if page == "Overview":
    best_model = metrics_df.loc[metrics_df["r2"].idxmax()]
    lowest_mae = metrics_df.loc[metrics_df["mae"].idxmin()]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models Evaluated", len(metrics_df), "Regression models")
    col2.metric("Best R² Score", f"{best_model['r2'] * 100:.2f}%", best_model["model"])
    col3.metric("Lowest MAE", f"{lowest_mae['mae']:.2f}", lowest_mae["model"])
    col4.metric("Dataset", "4,063 households", "RECON-SL")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Model Performance")
        st.bar_chart(metrics_df.set_index("model")[["rmse"]])
    with right:
        st.subheader("Top Performing Model")
        st.success(f"**{best_model['model']}**")
        st.metric("R² Score", f"{best_model['r2'] * 100:.2f}%")
        st.write(f"MAE: **{best_model['mae']:.2f}**")
        st.write(f"RMSE: **{best_model['rmse']:.2f}**")

    st.subheader("Feature Importance")
    st.bar_chart(feature_df.set_index("feature")[["importance"]])

elif page == "Model Comparison":
    st.subheader("Ensemble Model Comparison")
    display_df = metrics_df.copy()
    display_df["r2"] = (display_df["r2"] * 100).round(2)
    display_df = display_df.rename(columns={"r2": "R² (%)"})
    st.dataframe(
        display_df.sort_values("RMSE") if "RMSE" in display_df else display_df.sort_values("rmse"),
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(display_df.set_index("model")[["rmse"]])

elif page == "Feature Analysis":
    st.subheader("Random Forest Feature Importance")
    st.bar_chart(feature_df.set_index("feature")[["importance"]])
    st.info("Historical consumption features account for approximately 98.09% of the total feature importance.")

else:
    st.subheader("Predict Household Consumption")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            consumption = st.number_input("Current month consumption", min_value=0.0, value=120.0, step=1.0)
            lag_1 = st.number_input("Previous month consumption", min_value=0.0, value=115.0, step=1.0)
            lag_2 = st.number_input("Two months ago", min_value=0.0, value=110.0, step=1.0)
        with col2:
            lag_3 = st.number_input("Three months ago", min_value=0.0, value=105.0, step=1.0)
            month_number = st.number_input("Month number", min_value=1, max_value=12, value=10)
            year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)

        submitted = st.form_submit_button("Predict Next Month")

    if submitted:
        values = {
            "consumption": consumption,
            "consumption_lag_1": lag_1,
            "consumption_lag_2": lag_2,
            "consumption_lag_3": lag_3,
            "month_number": int(month_number),
            "year": int(year),
        }
        try:
            predictions = predict(model_bundle, values)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
        else:
            st.subheader("Prediction Results")
            for name, value in predictions.items():
                st.metric(name, f"{value:.2f}")
