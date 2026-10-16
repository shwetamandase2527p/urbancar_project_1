
import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="UrbanKart Customer Intelligence Hub",
    page_icon="🛍️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "best_regression_model.pkl"

DATASET_PATH = (
    PROJECT_DIR
    / "UrbanKart_HuggingFace_NLP_Customer_Reviews.csv"
)

SENTIMENT_MODEL = (
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)


# =========================================================
# 2. LOAD MODELS AND DATA
# =========================================================
@st.cache_resource
def load_regression_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_sentiment_model():
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL
    )


@st.cache_data
def load_dataset(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


def find_review_column(columns):
    candidates = [
        "review_text",
        "review",
        "customer_review",
        "feedback",
        "text"
    ]

    lookup = {
        str(col).strip().lower(): col
        for col in columns
    }

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    for col in columns:
        name = str(col).strip().lower()
        if "review" in name or "feedback" in name:
            return col

    return None


def predict_sentiments(classifier, texts, batch_size=16):
    predictions = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        output = classifier(
            batch,
            truncation=True,
            max_length=512,
            batch_size=batch_size
        )

        predictions.extend(output)

    labels = [
        str(item["label"]).replace("LABEL_", "Label ").lower()
        for item in predictions
    ]

    scores = [
        float(item["score"])
        for item in predictions
    ]

    return labels, scores


# =========================================================
# 3. MAIN PAGE HEADER
# =========================================================
st.title("🛍️ UrbanKart Customer Intelligence Hub")

st.write(
    "An integrated machine learning application for predicting "
    "annual customer revenue and analyzing customer feedback."
)

st.caption(
    "Regression Analytics | NLP Sentiment Analysis | Customer Insights"
)

st.divider()


# =========================================================
# 4. TWO FEATURES ON THE SAME PAGE
# =========================================================
revenue_tab, feedback_tab = st.tabs([
    "📊 Annual Revenue Prediction",
    "💬 Customer Feedback Analysis"
])


# =========================================================
# TAB 1: ANNUAL REVENUE PREDICTION
# =========================================================
with revenue_tab:

    st.header("Annual Revenue Prediction")

    st.write(
        "Enter customer details to estimate annual revenue "
        "using your trained regression model."
    )

    with st.form("revenue_form"):

        st.subheader("1. Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age", 18, 100, 35
            )

            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )

            city_tier = st.selectbox(
                "City Tier", ["Tier 1", "Tier 2", "Tier 3"]
            )

        with col2:
            income = st.number_input(
                "Annual Income (₹)",
                min_value=0.0,
                value=500000.0,
                step=10000.0
            )

            tenure = st.number_input(
                "Customer Tenure (Months)",
                1, 120, 24
            )

            acquisition_channel = st.selectbox(
                "Acquisition Channel",
                [
                    "Organic Search",
                    "Paid Search",
                    "Social Media",
                    "Affiliate",
                    "Offline Store"
                ]
            )

        with col3:
            preferred_category = st.selectbox(
                "Preferred Category",
                [
                    "Electronics",
                    "Fashion",
                    "Grocery",
                    "Books",
                    "Accessories"
                ]
            )

            preferred_channel = st.selectbox(
                "Preferred Channel",
                ["Website", "Mobile App", "Physical Store"]
            )

        st.divider()
        st.subheader("2. Customer Engagement")

        col1, col2, col3 = st.columns(3)

        with col1:
            website_visits = st.number_input(
                "Monthly Website Visits",
                min_value=0,
                value=10
            )

            app_sessions = st.number_input(
                "Monthly App Sessions",
                min_value=0,
                value=12
            )

        with col2:
            marketing_response = st.slider(
                "Marketing Response Rate",
                0.0, 1.0, 0.25, 0.01
            )

            satisfaction = st.slider(
                "Customer Satisfaction Score",
                1.0, 5.0, 3.5, 0.1
            )

        with col3:
            churn = st.selectbox(
                "Customer Churn Status",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

            purchase_next_30_days = st.selectbox(
                "Purchase Expected in Next 30 Days?",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        st.divider()
        st.subheader("3. Purchase Behaviour")

        col1, col2, col3 = st.columns(3)

        with col1:
            purchase_frequency = st.number_input(
                "Monthly Purchase Frequency",
                min_value=0.0,
                value=0.5,
                step=0.1
            )

            average_order_value = st.number_input(
                "Average Order Value (₹)",
                min_value=0.0,
                value=1500.0,
                step=100.0
            )

            orders_last_12m = st.number_input(
                "Orders in Last 12 Months",
                min_value=0,
                value=6
            )

        with col2:
            discount_usage = st.slider(
                "Discount Usage Rate",
                0.0, 1.0, 0.40, 0.01
            )

            days_since_purchase = st.number_input(
                "Days Since Last Purchase",
                0, 365, 30
            )

            return_rate = st.slider(
                "Return Rate",
                0.0, 1.0, 0.10, 0.01
            )

        with col3:
            payment_delays = st.number_input(
                "Payment Delay Count",
                min_value=0,
                value=0
            )

            loyalty_points = st.number_input(
                "Loyalty Points",
                min_value=0,
                value=200
            )

            support_tickets = st.number_input(
                "Support Tickets in Last 12 Months",
                min_value=0,
                value=1
            )

        predict_button = st.form_submit_button(
            "Predict Annual Revenue",
            type="primary",
            use_container_width=True
        )

    if predict_button:

        input_data = {
            "Age": age,
            "Gender": gender,
            "City_Tier": city_tier,
            "Income": income,
            "Customer_Tenure_Months": tenure,
            "Acquisition_Channel": acquisition_channel,
            "Website_Visits_Monthly": website_visits,
            "App_Sessions_Monthly": app_sessions,
            "Purchase_Frequency_Monthly": purchase_frequency,
            "Average_Order_Value": average_order_value,
            "Discount_Usage_Rate": discount_usage,
            "Orders_Last_12M": orders_last_12m,
            "Days_Since_Last_Purchase": days_since_purchase,
            "Return_Rate": return_rate,
            "Support_Tickets_12M": support_tickets,
            "Customer_Satisfaction_Score": satisfaction,
            "Payment_Delay_Count": payment_delays,
            "Loyalty_Points": loyalty_points,
            "Marketing_Response_Rate": marketing_response,
            "Churn": churn,
            "Purchase_Next_30_Days": purchase_next_30_days,
            "Preferred_Category": preferred_category,
            "Preferred_Channel": preferred_channel
        }

        input_df = pd.DataFrame([input_data])

        try:
            with st.spinner("Loading model and predicting revenue..."):
                model = load_regression_model()
                prediction = float(model.predict(input_df)[0])

            st.divider()
            st.subheader("Revenue Prediction Result")

            st.metric(
                "Estimated Annual Revenue",
                f"₹{prediction:,.2f}"
            )

            st.success("Revenue prediction generated successfully.")

            with st.expander("View customer input details"):
                st.dataframe(
                    input_df.T.rename(
                        columns={0: "Input Value"}
                    ),
                    use_container_width=True
                )

            st.info(
                "This is a machine learning estimate, "
                "not a guarantee of actual revenue."
            )

        except Exception as exc:
            st.error(f"Revenue prediction failed: {exc}")


# =========================================================
# TAB 2: CUSTOMER FEEDBACK ANALYSIS
# =========================================================
with feedback_tab:

    st.header("Customer Feedback Analysis")

    st.write(
        "Analyze individual customer reviews or classify multiple "
        "reviews to understand customer sentiment."
    )

    st.caption(
        "Sentiment categories: positive, neutral and negative."
    )

    st.divider()

    # Load the dataset
    uploaded_file = st.file_uploader(
        "Upload a customer reviews CSV (optional)",
        type=["csv"],
        key="feedback_csv"
    )

    dataset = None
    source_name = None

    try:
        if uploaded_file is not None:
            dataset = load_dataset(uploaded_file.getvalue())
            source_name = uploaded_file.name

        elif DATASET_PATH.exists():
            dataset = load_dataset(DATASET_PATH.read_bytes())
            source_name = DATASET_PATH.name

    except Exception as exc:
        st.error(f"Could not read the dataset: {exc}")

    review_column = None

    if dataset is not None:
        review_column = find_review_column(dataset.columns)

        col1, col2, col3 = st.columns(3)

        col1.metric("Dataset Rows", f"{len(dataset):,}")
        col2.metric("Dataset Columns", f"{len(dataset.columns):,}")

        if review_column:
            empty_count = (
                dataset[review_column]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

            col3.metric("Empty Reviews", f"{empty_count:,}")

            st.success(
                f"Loaded {source_name}. "
                f"Review column: {review_column}"
            )
        else:
            col3.metric("Review Column", "Not found")
            st.warning(
                "No review-text column found. Use a column such as "
                "review_text, review or feedback."
            )

        with st.expander("Preview customer dataset"):
            st.dataframe(dataset.head(), use_container_width=True)

    else:
        st.info(
            "Place UrbanKart_HuggingFace_NLP_Customer_Reviews.csv "
            "in C:\\project or upload a CSV above."
        )

    st.divider()

    # Single review analysis
    st.subheader("1. Analyze an Individual Review")

    review_text = st.text_area(
        "Enter customer feedback",
        placeholder=(
            "Example: The product quality was excellent "
            "and delivery was very fast."
        ),
        height=120,
        key="single_review"
    )

    if st.button(
        "Analyze Review",
        type="primary",
        key="analyze_single"
    ):

        if not review_text.strip():
            st.warning("Please enter customer feedback first.")

        else:
            try:
                with st.spinner("Analyzing customer sentiment..."):
                    classifier = load_sentiment_model()

                    result = classifier(
                        review_text.strip(),
                        truncation=True,
                        max_length=512
                    )[0]

                label = str(result["label"]).lower()
                confidence = float(result["score"])

                st.subheader("Sentiment Result")

                col1, col2 = st.columns(2)

                with col1:
                    if "positive" in label:
                        st.success("Sentiment: Positive")
                    elif "negative" in label:
                        st.error("Sentiment: Negative")
                    else:
                        st.info(f"Sentiment: {label.title()}")

                with col2:
                    st.metric(
                        "Confidence Score",
                        f"{confidence:.1%}"
                    )

                st.progress(
                    max(0.0, min(1.0, confidence))
                )

            except Exception as exc:
                st.error(f"Sentiment analysis failed: {exc}")

                st.info(
                    "Check your internet connection and confirm "
                    "that transformers and torch are installed."
                )

    st.divider()

    # Batch review analysis
    st.subheader("2. Batch Customer Feedback Analysis")

    if dataset is not None and review_column is not None:

        row_limit = st.number_input(
            "Number of reviews to analyze",
            min_value=1,
            max_value=max(1, len(dataset)),
            value=min(100, max(1, len(dataset))),
            step=1
        )

        if st.button(
            "Run Batch Sentiment Analysis",
            key="batch_analysis",
            use_container_width=True
        ):

            try:
                batch_df = dataset.head(int(row_limit)).copy()

                reviews = (
                    batch_df[review_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                valid_mask = reviews.ne("")
                valid_texts = reviews[valid_mask].tolist()

                with st.spinner(
                    "Analyzing customer reviews. Please wait..."
                ):
                    classifier = load_sentiment_model()

                    if valid_texts:
                        labels, scores = predict_sentiments(
                            classifier,
                            valid_texts
                        )
                    else:
                        labels, scores = [], []

                batch_df["predicted_sentiment"] = (
                    "not analyzed (empty review)"
                )

                batch_df["sentiment_confidence"] = float("nan")

                batch_df.loc[
                    valid_mask, "predicted_sentiment"
                ] = labels

                batch_df.loc[
                    valid_mask, "sentiment_confidence"
                ] = scores

                st.session_state["urban_kart_results"] = batch_df

                st.success(
                    f"Analysis completed: {len(valid_texts)} "
                    "non-empty reviews analyzed."
                )

            except Exception as exc:
                st.error(f"Batch analysis failed: {exc}")

    elif dataset is None:
        st.info(
            "Load a customer reviews CSV to enable batch analysis."
        )

    # Show saved batch results
    results = st.session_state.get("urban_kart_results")

    if isinstance(results, pd.DataFrame):

        st.divider()
        st.subheader("3. Sentiment Analysis Results")

        valid_results = results[
            results["predicted_sentiment"]
            != "not analyzed (empty review)"
        ]

        if not valid_results.empty:

            distribution = (
                valid_results["predicted_sentiment"]
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Review Count")
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Sentiment Distribution**")
                st.bar_chart(
                    distribution.set_index("Sentiment")
                )

            with col2:
                st.markdown("**Sentiment Percentages**")

                percentages = (
                    valid_results["predicted_sentiment"]
                    .value_counts(normalize=True)
                    .mul(100)
                    .rename_axis("Sentiment")
                    .reset_index(name="Percentage")
                )

                percentages["Percentage"] = percentages[
                    "Percentage"
                ].map(lambda x: f"{x:.1f}%")

                st.dataframe(
                    percentages,
                    hide_index=True,
                    use_container_width=True
                )

        st.markdown("**Analyzed Reviews**")

        st.dataframe(
            results,
            use_container_width=True
        )

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Download Analysis Results as CSV",
            data=csv_data,
            file_name="UrbanKart_Customer_Feedback_Results.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.caption(
        "Note: Sentiment predictions are estimates. The pretrained "
        "model was trained on Twitter text, which may differ from "
        "your customers' reviews."
    )


# =========================================================
# 5. SHARED SIDEBAR
# =========================================================
with st.sidebar:

    st.title("🛍️ UrbanKart")

    st.write("Customer Intelligence Hub")

    st.divider()

    st.markdown("**Available Features**")
    st.write("📊 Annual Revenue Prediction")
    st.write("💬 Customer Feedback Analysis")

    st.divider()

    st.markdown("**Technology Stack**")
    st.write("• Python")
    st.write("• Streamlit")
    st.write("• Scikit-learn")
    st.write("• Hugging Face Transformers")

    st.caption(
        "Predictions and sentiment results should be validated "
        "before making business decisions."
    )
