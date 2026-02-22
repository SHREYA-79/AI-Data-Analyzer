import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

st.title("AI Data Analyzer")
st.write("Upload a CSV file and get automatic insights and visualizations.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Fix datatype issues
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    # Show preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Statistics
    st.subheader("Dataset Statistics")
    st.write(df.describe(include="all"))

    # Visualization
    st.subheader("Quick Visualization")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) > 0:

        column = st.selectbox("Select column for visualization", numeric_cols)

        st.bar_chart(df[column])

    else:
        st.write("No numeric columns available.")

    # Correlation heatmap
    if len(numeric_cols) > 1:

        st.subheader("Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots()

        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

        st.pyplot(fig)

    # AI insights
    st.subheader("AI Dataset Insights")

    if st.button("Analyze with AI"):

        with st.spinner("AI is analyzing the dataset..."):

            client = Groq(api_key=st.secrets["GROQ_API_KEY"])

            summary = df.describe(include="all").to_string()

            prompt = f"""
            Analyze this dataset and provide useful insights.

            Columns:
            {list(df.columns)}

            Summary statistics:
            {summary}

            Provide key insights, patterns, anomalies, and possible interpretations.
            """

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )

            st.success("Analysis Complete")

            st.write(response.choices[0].message.content)
