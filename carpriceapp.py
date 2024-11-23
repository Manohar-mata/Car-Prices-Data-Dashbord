import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Custom CSS
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    /* Main page styling */
    .stApp {
        background: linear-gradient(to bottom right, #EEF2FF, #E0E7FF);
    }
    
    /* Sidebar styling */
    .stSidebar {
        background-color: white;
        padding: 2rem;
        border-right: 1px solid #E5E7EB;
    }
    
    .stSidebar [data-testid="stSidebarNav"] {
        background-color: rgba(239, 246, 255, 0.5);
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1E40AF;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #4338CA;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 0.375rem;
        border: 1px solid #E5E7EB;
    }
    
    /* Plot styling */
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }
    
    /* Table styling */
    .dataframe {
        border: none !important;
    }
    
    .dataframe tbody tr:nth-child(odd) {
        background-color: #F9FAFB;
    }
    
    .dataframe tbody tr:hover {
        background-color: #F3F4F6;
    }
    
    /* Custom divider */
    .divider {
        height: 2px;
        background-color: #E5E7EB;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(path)

# Custom plot styling function
def style_plot(fig):
    sns.set_theme(style="whitegrid")
    fig.patch.set_facecolor('white')
    plt.style.use('seaborn')
    return fig

# Load the data
df = load_data()

# Numeric and categorical columns
numeric_df = df.select_dtypes(include=["float64", "int"])
categorical_columns = df.select_dtypes(include=["object"]).columns

# Sidebar navigation with custom styling
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: #1E40AF; font-weight: 600;'>Data Analysis</h2>
        <p style='color: #6B7280; font-size: 0.875rem;'>Dashboard Navigation</p>
    </div>
""", unsafe_allow_html=True)

options = st.sidebar.radio("", ["🏠 Home", "📊 Data Overview", "📈 Visualizations", "🔗 High Correlations"])

# Title and Home
if "🏠 Home" in options:
    st.markdown("""
        <div class='card'>
            <h1>Welcome to the Data Analysis Dashboard</h1>
            <p style='color: #6B7280; font-size: 1.1rem;'>
                This interactive dashboard allows you to explore and analyze your dataset through various visualizations 
                and statistical methods. Use the sidebar navigation to:
            </p>
            <ul style='color: #4B5563; margin-top: 1rem;'>
                <li>View basic data statistics and overview</li>
                <li>Create custom visualizations</li>
                <li>Analyze correlations between variables</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Data Overview
if "📊 Data Overview" in options:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>Data Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3>Dataset Preview</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(), height=300)
    
    with col2:
        st.markdown("<h3>Quick Statistics</h3>", unsafe_allow_html=True)
        st.markdown("""
            <div class='metric-container'>
                <p>Total Records: {}</p>
                <p>Number of Features: {}</p>
                <p>Numeric Features: {}</p>
                <p>Categorical Features: {}</p>
            </div>
        """.format(
            len(df),
            len(df.columns),
            len(numeric_df.columns),
            len(categorical_columns)
        ), unsafe_allow_html=True)
    
    st.markdown("<h3>Descriptive Statistics</h3>", unsafe_allow_html=True)
    st.dataframe(df.describe(), height=400)
    st.markdown("</div>", unsafe_allow_html=True)

# Visualizations
if "📈 Visualizations" in options:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>Data Visualizations</h2>", unsafe_allow_html=True)
    
    # Visualization controls
    col1, col2 = st.columns(2)
    with col1:
        vis_type = st.selectbox(
            "Select Visualization Type:",
            ["Scatterplot", "Line Plot", "Boxplot", "Pairplot"],
            key="vis_type"
        )
    
    with col2:
        selected_column = st.selectbox(
            "Select column to filter:",
            numeric_df.columns,
            key="filter_column"
        )
    
    # Filter range
    min_val, max_val = numeric_df[selected_column].min(), numeric_df[selected_column].max()
    range_filter = st.slider(
        f"Filter by {selected_column} range:",
        float(min_val),
        float(max_val),
        (float(min_val), float(max_val)),
        key="range_filter"
    )
    
    # Filter the dataset
    filtered_data = df[
        (df[selected_column] >= range_filter[0]) &
        (df[selected_column] <= range_filter[1])
    ]
    
    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    
    if vis_type == "Scatterplot":
        scatter_x = st.selectbox("Select X-axis variable:", numeric_df.columns)
        scatter_y = st.selectbox("Select Y-axis variable:", numeric_df.columns)
        if scatter_x and scatter_y:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=filtered_data, x=scatter_x, y=scatter_y, ax=ax)
            correlation, p_value = pearsonr(filtered_data[scatter_x], filtered_data[scatter_y])
            st.pyplot(style_plot(fig))
            
            st.markdown(f"""
                <div class='metric-container'>
                    <p><strong>Correlation Analysis:</strong></p>
                    <p>Pearson Correlation: {correlation:.2f}</p>
                    <p>P-value: {p_value:.2e}</p>
                </div>
            """, unsafe_allow_html=True)
    
    elif vis_type == "Line Plot":
        line_x = st.selectbox("Select X-axis variable:", numeric_df.columns)
        line_y = st.selectbox("Select Y-axis variable:", numeric_df.columns)
        if line_x and line_y:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=filtered_data, x=line_x, y=line_y, ax=ax)
            st.pyplot(style_plot(fig))
    
    elif vis_type == "Boxplot":
        box_x = st.selectbox("Select categorical variable (X-axis):", categorical_columns)
        box_y = st.selectbox("Select numeric variable (Y-axis):", numeric_df.columns)
        if box_x and box_y:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=filtered_data, x=box_x, y=box_y, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(style_plot(fig))
    
    elif vis_type == "Pairplot":
        selected_vars = st.multiselect(
            "Select variables for Pairplot:",
            numeric_df.columns,
            default=list(numeric_df.columns[:3])
        )
        if selected_vars:
            fig = sns.pairplot(data=filtered_data, vars=selected_vars)
            st.pyplot(style_plot(fig.fig))
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# High Correlations
if "🔗 High Correlations" in options:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>Correlation Analysis</h2>", unsafe_allow_html=True)
    
    # Calculate correlations
    correlation_matrix = numeric_df.corr()
    
    # Get high correlations
    high_correlations = (
        correlation_matrix.stack()
        .reset_index()
        .rename(columns={0: "Correlation", "level_0": "Variable 1", "level_1": "Variable 2"})
    )
    high_correlations = high_correlations[
        (high_correlations["Variable 1"] != high_correlations["Variable 2"]) &
        (high_correlations["Correlation"] > 0.5)
    ]
    high_correlations = high_correlations.drop_duplicates(subset=["Correlation"])
    
    # Display correlations
    correlated_columns = list(
        set(high_correlations["Variable 1"]).union(set(high_correlations["Variable 2"]))
    )
    
    if correlated_columns:
        filtered_corr_matrix = correlation_matrix.loc[correlated_columns, correlated_columns]
        
        st.markdown("<h3>Correlation Heatmap</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            filtered_corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )
        plt.title("High Correlation Heatmap (> 0.5)")
        st.pyplot(style_plot(fig))
        
        st.markdown("<h3>Correlation Details</h3>", unsafe_allow_html=True)
        st.dataframe(
            high_correlations.sort_values("Correlation", ascending=False),
            height=400
        )
    else:
        st.info("No correlations greater than 0.5 found in the dataset.")
    
    st.markdown("</div>", unsafe_allow_html=True)
