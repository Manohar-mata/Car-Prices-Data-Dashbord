import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load Data
@st.cache_data
def load_data():
    path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(path)

# Custom plot styling function
def style_plot(fig):
    sns.set_theme(style="whitegrid")
    fig.patch.set_facecolor('#f5f5f5')
    return fig

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Data Overview", "Visualizations"])

# Load the data
df = load_data()

# Title and Home
if options == "Home":
    st.title("Data Analysis Dashboard")
    st.write("Welcome to the Data Analysis Dashboard! Use the sidebar to explore the dataset and visualize relationships.")

# Data Overview
if options == "Data Overview":
    st.header("Data Overview")
    st.write("### Dataset Preview")
    st.write(df.head())
    st.write("### Descriptive Statistics")
    st.write(df.describe())

# Visualizations
if options == "Visualizations":
    st.sidebar.subheader("Visualization Options")
    vis_type = st.sidebar.radio(
        "Select Visualization Type:",
        ["Scatterplot", "Line Plot", "Boxplot", "Violin Plot", "Pairplot"]
    )

    numeric_df = df.select_dtypes(include=["float64", "int"])
    categorical_columns = df.select_dtypes(include=["object"]).columns

    # Scatterplot
    if vis_type == "Scatterplot":
        st.write("### Scatterplot with P-Value")
        scatter_x = st.selectbox("Select X-axis variable:", numeric_df.columns)
        scatter_y = st.selectbox("Select Y-axis variable:", numeric_df.columns)
        if scatter_x and scatter_y:
            fig, ax = plt.subplots()
            sns.scatterplot(data=numeric_df, x=scatter_x, y=scatter_y, ax=ax)
            correlation, p_value = pearsonr(numeric_df[scatter_x], numeric_df[scatter_y])
            st.write(f"**Pearson Correlation**: {correlation:.2f}")
            st.write(f"**P-value**: {p_value:.2e}")
            st.pyplot(style_plot(fig))

    # Line Plot
    elif vis_type == "Line Plot":
        st.write("### Line Plot")
        line_x = st.selectbox("Select X-axis variable:", numeric_df.columns)
        line_y = st.selectbox("Select Y-axis variable:", numeric_df.columns)
        if line_x and line_y:
            fig, ax = plt.subplots()
            sns.lineplot(data=numeric_df, x=line_x, y=line_y, ax=ax)
            st.pyplot(style_plot(fig))

    # Boxplot
    elif vis_type == "Boxplot":
        st.write("### Boxplot")
        box_x = st.selectbox("Select categorical variable (X-axis):", categorical_columns)
        box_y = st.selectbox("Select numeric variable (Y-axis):", numeric_df.columns)
        if box_x and box_y:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=box_x, y=box_y, ax=ax)
            st.pyplot(style_plot(fig))

    # Violin Plot
    elif vis_type == "Violin Plot":
        st.write("### Violin Plot")
        violin_x = st.selectbox("Select categorical variable (X-axis):", categorical_columns)
        violin_y = st.selectbox("Select numeric variable (Y-axis):", numeric_df.columns)
        if violin_x and violin_y:
            fig, ax = plt.subplots()
            sns.violinplot(data=df, x=violin_x, y=violin_y, ax=ax)
            st.pyplot(style_plot(fig))

    # Pairplot
    elif vis_type == "Pairplot":
        st.write("### Pairplot")
        selected_vars = st.multiselect("Select variables for Pairplot:", numeric_df.columns, default=numeric_df.columns[:3])
        if selected_vars:
            fig = sns.pairplot(data=numeric_df, vars=selected_vars)
            st.pyplot(style_plot(fig.fig))
