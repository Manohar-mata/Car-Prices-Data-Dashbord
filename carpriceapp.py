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

# Load the data
df = load_data()

# Define numeric and categorical columns globally
numeric_df = df.select_dtypes(include=["float64", "int"])
categorical_columns = df.select_dtypes(include=["object"]).columns

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Data Overview", "Visualizations", "Data Distribution", "Correlation Heatmap"])

# Home Section
if options == "Home":
    st.title("Data Analysis Dashboard")
    st.write("Welcome to the Data Analysis Dashboard!")
    st.write("Use the sidebar to explore the dataset, visualize relationships, and uncover insights.")

# Data Overview Section
if options == "Data Overview":
    st.header("Data Overview")
    st.write("### Dataset Preview")
    st.write(df.head())
    st.write("### Descriptive Statistics")
    st.write(df.describe())

# Visualizations Section
if options == "Visualizations":
    st.sidebar.subheader("Visualization Options")
    vis_type = st.sidebar.radio(
        "Select Visualization Type:",
        ["Scatterplot", "Line Plot", "Boxplot", "Pairplot"]
    )

    # Add slider to filter data by a numeric column
    st.sidebar.subheader("Data Filtering")
    selected_column = st.sidebar.selectbox("Select column to filter:", numeric_df.columns)
    min_val, max_val = numeric_df[selected_column].min(), numeric_df[selected_column].max()
    range_filter = st.sidebar.slider(f"Filter by {selected_column} range:", min_val, max_val, (min_val, max_val))

    # Filter the dataset based on the slider range
    filtered_data = df[(df[selected_column] >= range_filter[0]) & (df[selected_column] <= range_filter[1])]

    # Scatterplot
    if vis_type == "Scatterplot":
        st.write("### Scatterplot with P-Value")
        scatter_x = st.selectbox("Select X-axis variable:", numeric_df.columns)
        scatter_y = st.selectbox("Select Y-axis variable:", numeric_df.columns)
        if scatter_x and scatter_y:
            fig, ax = plt.subplots()
            sns.scatterplot(data=filtered_data, x=scatter_x, y=scatter_y, ax=ax)
            correlation, p_value = pearsonr(filtered_data[scatter_x], filtered_data[scatter_y])
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
            sns.lineplot(data=filtered_data, x=line_x, y=line_y, ax=ax)
            st.pyplot(style_plot(fig))

    # Boxplot
    elif vis_type == "Boxplot":
        st.write("### Boxplot")
        box_x = st.selectbox("Select categorical variable (X-axis):", categorical_columns)
        box_y = st.selectbox("Select numeric variable (Y-axis):", numeric_df.columns)
        if box_x and box_y:
            fig, ax = plt.subplots()
            sns.boxplot(data=filtered_data, x=box_x, y=box_y, ax=ax)
            st.pyplot(style_plot(fig))

    # Pairplot
    elif vis_type == "Pairplot":
        st.write("### Pairplot")
        selected_vars = st.multiselect("Select variables for Pairplot:", numeric_df.columns, default=numeric_df.columns[:3])
        if selected_vars:
            fig = sns.pairplot(data=filtered_data, vars=selected_vars)
            st.pyplot(style_plot(fig.fig))

# Data Distribution Section
if options == "Data Distribution":
    st.header("Data Distribution")
    dist_column = st.sidebar.selectbox("Select column for distribution:", numeric_df.columns)
    if dist_column:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=dist_column, kde=True, ax=ax)
        st.write(f"### Distribution of {dist_column}")
        st.pyplot(style_plot(fig))

# Correlation Heatmap Section
if options == "Correlation Heatmap":
    st.header("Correlation Heatmap")
    st.write("### Correlation Matrix")
    correlation_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(style_plot(fig))
