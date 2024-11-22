import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Car Data Analysis Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
    }
    .stPlot {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Data loading function with error handling
@st.cache_data
def load_data():
    try:
        path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Safe correlation calculation
def calculate_correlation(x, y):
    try:
        mask = ~(np.isinf(x) | np.isinf(y) | np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 2 or len(y_clean) < 2:
            return None, None
            
        return pearsonr(x_clean, y_clean)
    except Exception as e:
        st.warning(f"Could not calculate correlation: {str(e)}")
        return None, None

# Function to create and style plots
def style_plot(fig, title):
    plt.title(title, pad=20)
    plt.tight_layout()
    fig.set_facecolor('white')
    plt.style.use('seaborn')
    return fig

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Go to", ["Home", "Data Overview", "Visualizations"]
)

# Load data
df = load_data()

if df is None:
    st.error("Failed to load data. Please check the data source and try again.")
    st.stop()

# Home page
if options == "Home":
    st.title("Car Data Analysis Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            ### 📊 Dashboard Features
            - Interactive data exploration
            - Multiple visualization types
            - Statistical analysis
            - Real-time correlations
        """)
    
    with col2:
        st.markdown(f"""
            ### 📈 Dataset Overview
            - Total Records: {len(df)}
            - Features: {len(df.columns)}
            - Numeric Features: {len(df.select_dtypes(include=['float64', 'int64']).columns)}
            - Categorical Features: {len(df.select_dtypes(include=['object']).columns)}
        """)

# Data Overview section
elif options == "Data Overview":
    st.title("Data Overview")
    
    tab1, tab2 = st.tabs(["Data Preview", "Statistics"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
        
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)

# Visualizations section
elif options == "Visualizations":
    st.title("Data Visualizations")
    
    # Filter numeric and categorical columns
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    categorical_columns = df.select_dtypes(include=["object"]).columns
    
    # Visualization type selector
    vis_type = st.sidebar.radio(
        "Select Visualization Type:",
        ["Scatterplot", "Line Plot", "Boxplot", "Violin Plot", "Pairplot"]
    )
    
    # Set figure size
    plt.figure(figsize=(10, 6))
    
    # Scatterplot
    if vis_type == "Scatterplot":
        st.subheader("Scatterplot Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            scatter_x = st.selectbox("Select X-axis variable:", numeric_df.columns)
        with col2:
            scatter_y = st.selectbox("Select Y-axis variable:", numeric_df.columns)
            
        if scatter_x and scatter_y:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=numeric_df, x=scatter_x, y=scatter_y, ax=ax)
            
            # Calculate correlation
            correlation, p_value = calculate_correlation(numeric_df[scatter_x], numeric_df[scatter_y])
            
            if correlation is not None:
                st.write(f"**Pearson Correlation**: {correlation:.2f}")
                st.write(f"**P-value**: {p_value:.2e}")
            
            st.pyplot(style_plot(fig, f"{scatter_x} vs {scatter_y}"))
    
    # Line Plot
    elif vis_type == "Line Plot":
        st.subheader("Line Plot Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            line_x = st.selectbox("Select X-axis variable:", numeric_df.columns)
        with col2:
            line_y = st.selectbox("Select Y-axis variable:", numeric_df.columns)
            
        if line_x and line_y:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=numeric_df, x=line_x, y=line_y, ax=ax)
            st.pyplot(style_plot(fig, f"Trend: {line_x} vs {line_y}"))
    
    # Boxplot
    elif vis_type == "Boxplot":
        st.subheader("Boxplot Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            box_x = st.selectbox("Select categorical variable:", categorical_columns)
        with col2:
            box_y = st.selectbox("Select numeric variable:", numeric_df.columns)
            
        if box_x and box_y:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x=box_x, y=box_y, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(style_plot(fig, f"Distribution of {box_y} by {box_x}"))
    
    # Violin Plot
    elif vis_type == "Violin Plot":
        st.subheader("Violin Plot Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            violin_x = st.selectbox("Select categorical variable:", categorical_columns)
        with col2:
            violin_y = st.selectbox("Select numeric variable:", numeric_df.columns)
            
        if violin_x and violin_y:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=df, x=violin_x, y=violin_y, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(style_plot(fig, f"Distribution of {violin_y} by {violin_x}"))
    
    # Pairplot
    elif vis_type == "Pairplot":
        st.subheader("Pairplot Analysis")
        
        selected_vars = st.multiselect(
            "Select variables (max 4 recommended):",
            numeric_df.columns,
            default=list(numeric_df.columns[:3])
        )
        
        if selected_vars:
            if len(selected_vars) > 4:
                st.warning("Selecting too many variables may affect performance.")
            
            with st.spinner("Generating pairplot..."):
                fig = sns.pairplot(data=df[selected_vars])
                st.pyplot(fig)

# Footer
st.markdown("""
    ---
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Car Data Analysis Dashboard | Created with Streamlit
    </div>
""", unsafe_allow_html=True)
