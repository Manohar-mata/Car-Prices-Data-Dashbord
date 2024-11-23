import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #f8f9fa, #dce6f5);
        font-family: 'Arial', sans-serif;
    }
    
    h1, h2, h3 {
        color: #002147;
        font-weight: bold;
    }

    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv"
    return pd.read_csv(url)

# Load data
df = load_data()
numeric_df = df.select_dtypes(include=["float64", "int"])
categorical_columns = df.select_dtypes(include=["object"]).columns

# Sidebar navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("", ["Home", "Data Overview", "Visualizations", "High Correlations"])

# Home Page
if options == "Home":
    st.title("Welcome to the Data Analysis Dashboard")
    st.write("Explore your dataset interactively using visualizations, statistical insights, and correlations.")

# Data Overview Section
elif options == "Data Overview":
    st.header("Data Overview")
    st.dataframe(df.head())
    st.write(f"**Total Records:** {len(df)}")
    st.write(f"**Features:** {len(df.columns)}")
    st.write(f"**Numeric Features:** {len(numeric_df.columns)}")
    st.write(f"**Categorical Features:** {len(categorical_columns)}")

# Visualizations Section
elif options == "Visualizations":
    st.header("Visualizations")
    vis_type = st.selectbox("Select Visualization Type:", ["Scatterplot", "Boxplot"])
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X-axis Variable:", numeric_df.columns)
    with col2:
        y_var = st.selectbox("Y-axis Variable:", numeric_df.columns)
    
    if vis_type == "Scatterplot":
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
        st.pyplot(fig)

# High Correlations Section
elif options == "High Correlations":
    st.header("High Correlations")
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
