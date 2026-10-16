
import streamlit as st
import pandas as pd
from transformers import pipeline
from pathlib import Path

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="UrbanKart | Customer Feedback Analysis",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Customer Feedback Analysis")
st.caption(
    "Analyze customer opinions using Hugging Face sentiment analysis."
)

# -------------------------------
# File and model configuration
# -------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Works in a Streamlit .py file and can also be tested in a notebook
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

CSV_FILE = BASE_DIR / "UrbanKart_HuggingFace_NLP_Customer Reviews.csv"


# -------------------------------
# Load sentiment model
# -------------------------------
@st.cache_resource(show_spinner="Loading sentiment analysis model...")
def load_sentiment_model():
    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME
    )


# -------------------------------
# Analyze reviews in batches
# -------------------------------
def analyze_reviews(reviews, sentiment_model, batch_size=16):
    all_predictions = []

    for start in range(0, len(reviews), batch_size):
        batch = reviews[start:start + batch_size]

        predictions = sentiment_model(
            batch,
            batch_size=batch_size,
            truncation=True,
            max_length=512
        )

        all_predictions.extend(predictions)

    results = pd.DataFrame({
        "Predicted_Sentiment": [
            str(item["label"]).strip().capitalize()
            for item in all_predictions
        ],
        "Confidence_Score": [
            round(float(item["score"]) * 100, 2)
            for item in all_predictions
        ]
    })

    return results


# -------------------------------
# Individual review analysis
# -------------------------------
st.subheader("1. Analyze an Individual Review")

review_text = st.text_area(
    "Enter customer review",
    placeholder="Example: The product quality is excellent!",
    height=120
)

if st.button("Analyze Review", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a customer review.")

    else:
        try:
            model = load_sentiment_model()

            with st.spinner("Analyzing review..."):
                result = model(
                    review_text.strip(),
                    truncation=True,
                    max_length=512
                )[0]

            sentiment = str(result["label"]).strip().capitalize()
            confidence = float(result["score"]) * 100

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Sentiment", sentiment)

            with col2:
                st.metric(
                    "Confidence Score",
                    f"{confidence:.2f}%"
                )

        except Exception as error:
            st.error(f"Unable to analyze review: {error}")

st.divider()


# -------------------------------
# Batch customer review analysis
# -------------------------------
st.subheader("2. Batch Analysis of Customer Reviews")

st.caption(f"Expected dataset location: {CSV_FILE}")

if st.button("Load Dataset and Analyze All Reviews"):
    try:
        if not CSV_FILE.is_file():
            st.error(
                "Dataset not found. Place "
                "'HuggingFace_NLP_Customer Reviews.csv' "
                f"in this folder: {BASE_DIR}"
            )

        else:
            df = pd.read_csv(CSV_FILE)

            # Validate required column
            if "review_text" not in df.columns:
                st.error(
                    "Missing required column: 'review_text'. "
                    f"Available columns: {df.columns.tolist()}"
                )

            else:
                # Identify non-empty reviews
                valid_mask = (
                    df["review_text"].notna()
                    & df["review_text"].astype(str).str.strip().ne("")
                )

                valid_reviews = (
                    df.loc[valid_mask, "review_text"]
                    .astype(str)
                    .tolist()
                )

                if not valid_reviews:
                    st.warning("No valid customer reviews were found.")

                else:
                    model = load_sentiment_model()

                    progress = st.progress(0)
                    status = st.empty()

                    # Process data in chunks
                    chunk_size = 256
                    analyzed_parts = []

                    for start in range(0, len(valid_reviews), chunk_size):
                        batch = valid_reviews[
                            start:start + chunk_size
                        ]

                        analyzed_parts.append(
                            analyze_reviews(batch, model)
                        )

                        completed = min(
                            start + chunk_size,
                            len(valid_reviews)
                        )

                        progress.progress(
                            completed / len(valid_reviews)
                        )

                        status.text(
                            f"Analyzed {completed:,} of "
                            f"{len(valid_reviews):,} reviews"
                        )

                    analyzed = pd.concat(
                        analyzed_parts,
                        ignore_index=True
                    )

                    # Preserve all original rows
                    output_df = df.copy()

                    output_df["Predicted_Sentiment"] = pd.NA
                    output_df["Confidence_Score"] = pd.NA

                    output_df.loc[
                        valid_mask, "Predicted_Sentiment"
                    ] = analyzed["Predicted_Sentiment"].to_numpy()

                    output_df.loc[
                        valid_mask, "Confidence_Score"
                    ] = analyzed["Confidence_Score"].to_numpy()

                    st.session_state["feedback_results"] = output_df

                    status.success("Sentiment analysis completed!")

    except Exception as error:
        st.error(f"Batch analysis failed: {error}")


# -------------------------------
# Display batch results
# -------------------------------
if "feedback_results" in st.session_state:
    output_df = st.session_state["feedback_results"]

    st.subheader("3. Sentiment Analysis Results")

    valid_results = output_df.dropna(
        subset=["Predicted_Sentiment"]
    )

    counts = valid_results["Predicted_Sentiment"].value_counts()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Reviews Analyzed", len(valid_results))

    with col2:
        st.metric(
            "Positive Reviews",
            int(counts.get("Positive", 0))
        )

    with col3:
        st.metric(
            "Negative Reviews",
            int(counts.get("Negative", 0))
        )

    st.subheader("Sentiment Distribution")

    distribution = (
        valid_results["Predicted_Sentiment"]
        .value_counts()
        .reindex(
            ["Positive", "Neutral", "Negative"],
            fill_value=0
        )
    )

    chart_df = distribution.rename_axis(
        "Sentiment"
    ).reset_index(name="Number of Reviews")

    st.bar_chart(
        chart_df,
        x="Sentiment",
        y="Number of Reviews"
    )

    st.subheader("Analyzed Customer Reviews")

    st.dataframe(
        output_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = output_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Download Analyzed Reviews CSV",
        data=csv_data,
        file_name="UrbanKart_Customer_Feedback_Analysis.csv",
        mime="text/csv"
    )
