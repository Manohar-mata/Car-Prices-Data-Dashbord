import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Data Overview", "Visualizations", "Statistics"])

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
    buffer = []
    df.info(buf=buffer)
    st.text("\n".join(buffer))  # Display `info()` output

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

    # Pairplot
    st.write("### Pairplot")
    selected_cols = st.multiselect("Select columns for pairplot:", numeric_df.columns)
    if selected_cols:
        sns.pairplot(df[selected_cols])
        st.pyplot()

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
