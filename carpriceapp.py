import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import io  # For capturing df.info() output

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Go to", ["Home", "Data Overview", "Visualizations", "Statistics"]
)

# Title and introduction
if options == "Home":
    st.title("Data Analysis Dashboard")
    st.write("This app visualizes data and provides insights. Navigate using the sidebar.")

# Load data
@st.cache_data
def load_data():
    path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(path)

df = load_data()

if options == "Data Overview":
    st.header("Data Overview")
    st.write("### Dataset Preview")
    st.write(df.head())

    st.write("### Basic Information")
    buffer = io.StringIO()  # Create a buffer to capture df.info()
    df.info(buf=buffer)  # Write df.info() output to the buffer
    info_text = buffer.getvalue()  # Extract buffer content as a string
    st.text(info_text)  # Display the captured output

    st.write("### Descriptive Statistics")
    st.write(df.describe())

if options == "Visualizations":
    st.header("Visualizations")

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int'])  # Select numeric columns only
    if not numeric_df.empty:
        numeric_df = numeric_df.dropna()  # Drop rows with missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric data available for correlation heatmap.")

    # Scatterplot
    st.write("### Scatterplot with p-value")
    scatter_x = st.selectbox("Select X-axis variable for scatterplot:", numeric_df.columns)
    scatter_y = st.selectbox("Select Y-axis variable for scatterplot:", numeric_df.columns)
    if scatter_x and scatter_y:
        fig, ax = plt.subplots()
        sns.scatterplot(data=numeric_df, x=scatter_x, y=scatter_y, ax=ax)
        # Calculate and display Pearson correlation and p-value
        correlation, p_value = pearsonr(numeric_df[scatter_x], numeric_df[scatter_y])
        st.write(f"**Pearson Correlation**: {correlation:.2f}")
        st.write(f"**P-value**: {p_value:.2e}")
        st.pyplot(fig)

    # Regression Plot
    st.write("### Regression Plot")
    reg_x = st.selectbox("Select X-axis variable for regression plot:", numeric_df.columns)
    reg_y = st.selectbox("Select Y-axis variable for regression plot:", numeric_df.columns)
    if reg_x and reg_y:
        fig, ax = plt.subplots()
        sns.regplot(data=numeric_df, x=reg_x, y=reg_y, ax=ax, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
        st.pyplot(fig)

    # Line Plot
    st.write("### Line Plot")
    line_x = st.selectbox("Select X-axis variable for line plot:", numeric_df.columns)
    line_y = st.selectbox("Select Y-axis variable for line plot:", numeric_df.columns)
    if line_x and line_y:
        fig, ax = plt.subplots()
        sns.lineplot(data=numeric_df, x=line_x, y=line_y, ax=ax)
        st.pyplot(fig)

    # Boxplot
    st.write("### Boxplot for Categorical Variables")
    categorical_columns = df.select_dtypes(include=["object"]).columns
    box_x = st.selectbox("Select categorical variable for boxplot (X-axis):", categorical_columns)
    box_y = st.selectbox("Select numeric variable for boxplot (Y-axis):", numeric_df.columns)
    if box_x and box_y:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=box_x, y=box_y, ax=ax)
        st.pyplot(fig)

if options == "Statistics":
    st.header("Statistics and Grouping")
    
    # Grouped Means
    st.write("### Grouped Means")
    group_column = st.selectbox("Select column to group by:", df.columns)
    if group_column:
        grouped_means = df.groupby(group_column).mean()
        st.write(grouped_means)

    # Value Counts
    st.write("### Value Counts")
    value_counts_col = st.selectbox("Select column for value counts:", df.columns)
    if value_counts_col:
        st.bar_chart(df[value_counts_col].value_counts())
