import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Go to", ["Home", "Data Overview", "Visualizations"]
)

# Load data
@st.cache_data
def load_data():
    path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(path)

df = load_data()

# Title and Introduction
if options == "Home":
    st.title("Data Analysis Dashboard")
    st.write("Welcome to the Data Analysis Dashboard! Use the sidebar to explore the dataset and visualize relationships.")

# Data Overview Section
if options == "Data Overview":
    st.header("Data Overview")
    st.write("### Dataset Preview")
    st.write(df.head())

    st.write("### Descriptive Statistics")
    st.write(df.describe())

# Visualizations Section
if options == "Visualizations":
    st.sidebar.subheader("Visualization Types")
    vis_type = st.sidebar.radio(
        "Select Visualization Type:",
        ["Scatterplot", "Line Plot", "Boxplot", "Violin Plot", "Pairplot"]
    )

    numeric_df = df.select_dtypes(include=["float64", "int"])  # Filter numeric columns
    categorical_columns = df.select_dtypes(include=["object"]).columns  # Filter categorical columns

    # Scatterplot
    if vis_type == "Scatterplot":
        st.write("### Scatterplot with p-value")
        scatter_x = st.selectbox("Select X-axis variable:", numeric_df.columns)
        scatter_y = st.selectbox("Select Y-axis variable:", numeric_df.columns)
        if scatter_x and scatter_y:
            fig, ax = plt.subplots()
            sns.scatterplot(data=numeric_df, x=scatter_x, y=scatter_y, ax=ax)
            # Calculate and display Pearson correlation and p-value
            correlation, p_value = pearsonr(numeric_df[scatter_x], numeric_df[scatter_y])
            st.write(f"**Pearson Correlation**: {correlation:.2f}")
            st.write(f"**P-value**: {p_value:.2e}")
            st.pyplot(fig)

    # Line Plot
    elif vis_type == "Line Plot":
        st.write("### Line Plot")
        line_x = st.selectbox("Select X-axis variable:", numeric_df.columns)
        line_y = st.selectbox("Select Y-axis variable:", numeric_df.columns)
        if line_x and line_y:
            fig, ax = plt.subplots()
            sns.lineplot(data=numeric_df, x=line_x, y=line_y, ax=ax)
            st.pyplot(fig)

    # Boxplot
    elif vis_type == "Boxplot":
        st.write("### Boxplot for Categorical Variables")
        box_x = st.selectbox("Select categorical variable (X-axis):", categorical_columns)
        box_y = st.selectbox("Select numeric variable (Y-axis):", numeric_df.columns)
        if box_x and box_y:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=box_x, y=box_y, ax=ax)
            st.pyplot(fig)

    # Violin Plot
    elif vis_type == "Violin Plot":
        st.write("### Violin Plot for Categorical Variables")
        violin_x = st.selectbox("Select categorical variable (X-axis):", categorical_columns)
        violin_y = st.selectbox("Select numeric variable (Y-axis):", numeric_df.columns)
        if violin_x and violin_y:
            fig, ax = plt.subplots()
            sns.violinplot(data=df, x=violin_x, y=violin_y, ax=ax)
            st.pyplot(fig)

    # Pairplot
    elif vis_type == "Pairplot":
        st.write("### Pairplot of Numeric Variables")
        selected_vars = st.multiselect("Select variables to include in pairplot:", numeric_df.columns, default=numeric_df.columns[:3])
        if selected_vars:
            fig = sns.pairplot(data=df, vars=selected_vars)
            st.pyplot(fig)
