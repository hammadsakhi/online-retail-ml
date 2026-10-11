import requests
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Online Retail ML Dashboard",
    page_icon="🛍️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "customer_features.csv"

SUPERVISED_MODEL_PATH = MODEL_DIR / "repeat_purchase_model.pkl"
SUPERVISED_FEATURES_PATH = MODEL_DIR / "repeat_purchase_features.pkl"
KMEANS_MODEL_PATH = MODEL_DIR / "kmeans_model.pkl"
SCALER_PATH = MODEL_DIR / "kmeans_scaler.pkl"

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "UniqueProducts"
]
API_URL = "http://127.0.0.1:8000"
CLUSTER_NAMES = {
    0: "Regular Buyers",
    1: "Valuable Repeat Buyers",
    2: "High-Value Customers",
    3: "Inactive / At-Risk Customers"
}

MODEL_RESULTS = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": 0.667656,
        "Precision": 0.720994,
        "Recall": 0.679688,
        "F1-Score": 0.699732,
        "ROC-AUC": 0.728655,
        "PR-AUC": 0.796082
    },
    {
        "Model": "Decision Tree",
        "Accuracy": 0.658754,
        "Precision": 0.709239,
        "Recall": 0.679688,
        "F1-Score": 0.694149,
        "ROC-AUC": 0.718674,
        "PR-AUC": 0.769870
    },
    {
        "Model": "Random Forest",
        "Accuracy": 0.679525,
        "Precision": 0.770968,
        "Recall": 0.622396,
        "F1-Score": 0.688761,
        "ROC-AUC": 0.735623,
        "PR-AUC": 0.808776
    },
    {
        "Model": "Gradient Boosting",
        "Accuracy": 0.660237,
        "Precision": 0.705570,
        "Recall": 0.692708,
        "F1-Score": 0.699080,
        "ROC-AUC": 0.733908,
        "PR-AUC": 0.807336
    }
])

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 2. LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    required_files = [
        SUPERVISED_MODEL_PATH,
        SUPERVISED_FEATURES_PATH,
        KMEANS_MODEL_PATH,
        SCALER_PATH
    ]

    missing_files = [
        str(path) for path in required_files if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing_files)
        )

    supervised_model = joblib.load(SUPERVISED_MODEL_PATH)
    supervised_features = joblib.load(SUPERVISED_FEATURES_PATH)
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return (
        supervised_model,
        supervised_features,
        kmeans_model,
        scaler
    )


# Only load pickle files that you created or otherwise trust.
try:
    (
        supervised_model,
        supervised_features,
        kmeans_model,
        scaler
    ) = load_models()

except Exception as error:
    st.error(f"Could not load the saved models: {error}")
    st.info(
        "Check that all four model files exist in your models folder."
    )
    st.stop()


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def prepare_customer_data(data):
    """Validate customer features before prediction."""
    missing = [col for col in FEATURES if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.copy()

    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data[FEATURES].isna().any().any():
        raise ValueError(
            "Customer features contain missing or non-numeric values."
        )

    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Customer features contain infinite values.")

    if (data[FEATURES] < 0).any().any():
        raise ValueError(
            "Customer purchasing features cannot be negative."
        )

    return data


def segment_customers(data):
    """Apply the saved scaler and K-Means model."""
    data = prepare_customer_data(data)

    scaled_features = scaler.transform(data[FEATURES])
    data["Cluster"] = kmeans_model.predict(scaled_features)

    data["Segment"] = data["Cluster"].map(
        CLUSTER_NAMES
    ).fillna(
        data["Cluster"].map(lambda cluster: f"Cluster {cluster}")
    )

    return data


def load_local_customer_data():
    """Load the optional local customer CSV."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    return None


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🛍️ Online Retail ML")
st.sidebar.caption("Machine Learning Project")


page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard Overview",
        "Repeat-Purchase Prediction",
        "Customer Segmentation",
        "Model Performance",
        "Prediction History"
    ]
)


st.sidebar.divider()
st.sidebar.write("**Models used**")
st.sidebar.write("• Random Forest Classifier")
st.sidebar.write("• K-Means Clustering")


# --------------------------------------------------
# 5. DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.markdown(
        '<p class="main-title">Online Retail Dashboard</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore customer purchasing behavior using supervised "
        "and unsupervised machine learning."
    )
    
    st.subheader("Live Prediction Statistics")

    try:
        response = requests.get(
            f"{API_URL}/prediction-stats",
            timeout=15
        )
        response.raise_for_status()

        stats = response.json()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Predictions",
            stats["total_predictions"]
        )

        c2.metric(
            "Predicted Repeat Purchases",
            stats["repeat_predictions"]
        )

        c3.metric(
            "Predicted Non-Repeats",
            stats["non_repeat_predictions"]
        )

        st.subheader("Prediction Distribution")

        chart_data = pd.DataFrame({
            "Prediction Type": [
                "Repeat Purchase",
                "Non-Repeat Purchase"
            ],
            "Count": [
                stats["repeat_predictions"],
                stats["non_repeat_predictions"]
            ]
        })

        st.bar_chart(
            chart_data.set_index("Prediction Type"),
            y="Count"
        )

        st.subheader("10 Most Recent Predictions")

        recent = stats.get("recent_predictions", [])

        if recent:
            recent_df = pd.DataFrame(recent)

            st.dataframe(
                recent_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No prediction history is available yet.")

    except requests.exceptions.RequestException as error:
        st.warning(
            f"Live prediction statistics unavailable: {error}"
        )

    uploaded = st.file_uploader(
        "Upload customer_features.csv (optional)",
        type=["csv"],
        key="overview_csv"
    )

    try:
        if uploaded is not None:
            customers = pd.read_csv(uploaded)
        else:
            customers = load_local_customer_data()

        if customers is None:
            st.info(
                "Upload customer_features.csv to view customer "
                "statistics. You can still use the other pages."
            )
        else:
            customers = prepare_customer_data(customers)

            total_customers = len(customers)
            avg_spending = customers["Monetary"].mean()
            avg_frequency = customers["Frequency"].mean()
            avg_recency = customers["Recency"].mean()

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Customers", f"{total_customers:,}")
            c2.metric("Average Spending", f"{avg_spending:,.2f}")
            c3.metric("Avg. Purchase Frequency", f"{avg_frequency:.2f}")
            c4.metric("Avg. Recency (days)", f"{avg_recency:.1f}")

            st.subheader("Customer Purchasing Statistics")

            left, right = st.columns(2)

            with left:
                st.write("Average Customer Spending")
                st.bar_chart(
                    customers["Monetary"].describe().iloc[1:2]
                )

            with right:
                st.write("Purchase Frequency Distribution")
                st.bar_chart(
                    customers["Frequency"]
                    .value_counts()
                    .sort_index()
                    .head(15)
                )

            if "RepeatPurchase" in customers.columns:
                st.subheader("Repeat-Purchase Distribution")

                target_counts = (
                    customers["RepeatPurchase"]
                    .value_counts()
                    .sort_index()
                )

                st.bar_chart(target_counts)

                st.caption(
                    "Target 1 = repeat purchase; target 0 = "
                    "not labelled as a repeat purchase in your dataset."
                )

            st.subheader("Customer Data Preview")
            st.dataframe(customers.head(100), use_container_width=True)

    except Exception as error:
        st.error(f"Could not display customer data: {error}")


# --------------------------------------------------
# 6. SUPERVISED LEARNING: PREDICTION
# --------------------------------------------------

elif page == "Repeat-Purchase Prediction":

    st.title("Repeat-Purchase Prediction")

    st.write(
        "Enter a customer's purchasing information. The saved "
        "Random Forest model will predict the target class."
    )

    st.caption(
        "Use the same feature definitions and units used when "
        "training your model."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)
        customer_id = st.number_input(
        "Customer ID",
        min_value=1,
        value=10001,
        step=1
        )
        with col1:
            recency = st.number_input(
                "Recency (days since last purchase)",
                min_value=0,
                value=30,
                step=1
            )

            frequency = st.number_input(
                "Frequency (purchase count)",
                min_value=0,
                value=5,
                step=1
            )

            monetary = st.number_input(
                "Monetary (total spending)",
                min_value=0.0,
                value=2500.0,
                step=100.0
            )

        with col2:
            total_quantity = st.number_input(
                "Total Quantity Purchased",
                min_value=0,
                value=500,
                step=10
            )

            unique_products = st.number_input(
                "Unique Products Purchased",
                min_value=0,
                value=25,
                step=1
            )

        submitted = st.form_submit_button(
            "Predict Repeat Purchase",
            type="primary",
            use_container_width=True
        )

    
    if submitted:

        payload = {
            "customer_id": int(customer_id),
            "Recency": int(recency),
            "Frequency": int(frequency),
            "Monetary": float(monetary),
            "TotalQuantity": int(total_quantity),
            "UniqueProducts": int(unique_products),
        }

        try:
            with st.spinner("Sending data to FastAPI..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=15
                )

            response.raise_for_status()
            result = response.json()

            prediction = int(
                result["predicted_repeat_purchase"]
            )
            probability = float(result["probability"])

            st.subheader("Prediction Result")

            if prediction == 1:
                st.success(
                    "Predicted class: Repeat purchase (1)"
                )
            else:
                st.warning(
                    "Predicted class: Not a repeat purchase (0)"
                )

            st.metric(
                "Estimated Probability of Repeat Purchase",
                f"{probability * 100:.2f}%"
            )

            st.progress(max(0.0, min(1.0, probability)))

            st.write("**Customer ID:**", result["customer_id"])
            st.write("**Prediction ID:**", result["prediction_id"])

            st.success(
                "Prediction received from FastAPI and saved "
                "to PostgreSQL."
            )

        except requests.exceptions.RequestException as error:
            st.error(f"Could not connect to FastAPI: {error}")

        except (KeyError, ValueError, TypeError) as error:
            st.error(f"Unexpected API response: {error}")


# --------------------------------------------------
# 7. UNSUPERVISED LEARNING: SEGMENTATION
# --------------------------------------------------

elif page == "Customer Segmentation":

    st.title("Customer Segmentation with K-Means")

    st.write(
        "The saved scaler and K-Means model assign each customer "
        "to a segment based on purchasing behavior."
    )

    uploaded = st.file_uploader(
        "Upload customer_features.csv",
        type=["csv"],
        key="segmentation_csv"
    )

    if uploaded is not None:

        try:
            customers = pd.read_csv(uploaded)
            segmented = segment_customers(customers)

            st.success("Customer segmentation completed.")

            c1, c2 = st.columns(2)
            c1.metric("Total Customers", f"{len(segmented):,}")
            c2.metric(
                "Segments Found",
                segmented["Cluster"].nunique()
            )

            st.subheader("Customers per Segment")

            counts = (
                segmented["Segment"]
                .value_counts()
                .rename_axis("Segment")
                .reset_index(name="Customers")
            )

            st.bar_chart(counts.set_index("Segment")["Customers"])

            st.subheader("Average Purchasing Behavior")

            summary = segmented.groupby("Segment")[FEATURES].mean()
            st.dataframe(summary.round(2), use_container_width=True)

            st.subheader("Cluster Distribution")

            fig, ax = plt.subplots()
            ax.scatter(
                segmented["Recency"],
                segmented["Monetary"],
                c=segmented["Cluster"],
                alpha=0.6
            )
            ax.set_xlabel("Recency (days)")
            ax.set_ylabel("Monetary (spending)")
            ax.set_title("Customers by K-Means Cluster")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Segmented Customer Records")
            st.dataframe(segmented, use_container_width=True)

            csv_data = segmented.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Segmented Customers",
                data=csv_data,
                file_name="segmented_customers.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Segmentation failed: {error}")

    else:
        st.info(
            "Upload customer_features.csv to assign customers "
            "to the saved K-Means clusters."
        )


# --------------------------------------------------
# 8. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "These supervised-model metrics are from your previously "
        "reported evaluation results."
    )

    st.dataframe(
        MODEL_RESULTS.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-Score": "{:.2%}",
            "ROC-AUC": "{:.3f}",
            "PR-AUC": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Compare Model Accuracy")

    accuracy_data = MODEL_RESULTS.set_index("Model")["Accuracy"]
    st.bar_chart(accuracy_data)

    st.subheader("Compare F1-Score")

    f1_data = MODEL_RESULTS.set_index("Model")["F1-Score"]
    st.bar_chart(f1_data)

    st.subheader("Compare ROC-AUC")

    auc_data = MODEL_RESULTS.set_index("Model")["ROC-AUC"]
    st.bar_chart(auc_data)

    best_model = MODEL_RESULTS.loc[
        MODEL_RESULTS["F1-Score"].idxmax()
    ]

    st.info(
        f"Highest F1-score in these results: "
        f"{best_model['Model']} "
        f"({best_model['F1-Score']:.2%})."
    )

    st.caption(
        "The displayed metrics are recorded evaluation results; "
        "the app does not retrain or reevaluate those models."
    )

# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------


elif page == "Prediction History":

    st.title("Prediction History")

    st.write(
        "Search saved ML predictions by customer ID."
    )

    with st.form("history_search_form"):
        customer_id_filter = st.number_input(
            "Customer ID (enter 0 to show all customers)",
            min_value=0,
            value=0,
            step=1
        )

        search_submitted = st.form_submit_button(
            "Search Predictions",
            type="primary"
        )

    try:
        params = {}

        if customer_id_filter > 0:
            params["customer_id"] = int(customer_id_filter)

        response = requests.get(
            f"{API_URL}/predictions",
            params=params,
            timeout=15
        )

        response.raise_for_status()
        predictions = response.json()

        if not predictions:
            st.info(
                "No prediction records found for this search."
            )

        else:
            history = pd.DataFrame(predictions)

            c1, c2, c3 = st.columns(3)

            c1.metric("Records Found", len(history))

            c2.metric(
                "Predicted Repeat",
                int(
                    (
                        history["predicted_repeat_purchase"] == 1
                    ).sum()
                )
            )

            c3.metric(
                "Predicted Non-Repeat",
                int(
                    (
                        history["predicted_repeat_purchase"] == 0
                    ).sum()
                )
            )

            st.subheader("Prediction Records")

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

            csv_data = history.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Search Results",
                data=csv_data,
                file_name="prediction_history.csv",
                mime="text/csv"
            )

    except requests.exceptions.RequestException as error:
        st.error(
            f"Could not retrieve prediction history: {error}"
        )

    except (ValueError, TypeError) as error:
        st.error(
            f"Could not read prediction data: {error}"
        )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("Online Retail ML Project")
st.sidebar.caption("Supervised + Unsupervised Learning")
