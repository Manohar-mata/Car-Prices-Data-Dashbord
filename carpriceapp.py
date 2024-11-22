import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem 3rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    h1 {
        color: #2c3e50;
        padding-bottom: 1rem;
        border-bottom: 2px solid #eee;
    }
    h2 {
        color: #34495e;
        margin-top: 2rem;
    }
    h3 {
        color: #7f8c8d;
    }
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 5px;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.image("https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/logo.png", width=100)
    st.title("Navigation")
    options = st.radio(
        "",
        ["🏠 Home", "📋 Data Overview", "📊 Visualizations", "📈 Statistics"],
        key="navigation"
    )

# Load data with error handling
@st.cache_data
def load_data():
    try:
        path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

if df is None:
    st.error("Failed to load data. Please check the data source and try again.")
    st.stop()

# Home page
if options == "🏠 Home":
    st.title("Data Analysis Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            ### 📌 Key Features
            - Interactive data exploration
            - Dynamic visualizations
            - Statistical analysis
            - Grouped insights
        """)
    
    with col2:
        st.markdown("""
            ### 🎯 Quick Stats
            - Records: {0}
            - Features: {1}
            - Numeric columns: {2}
            - Categorical columns: {3}
        """.format(
            len(df),
            len(df.columns),
            len(df.select_dtypes(include=['float64', 'int64']).columns),
            len(df.select_dtypes(include=['object']).columns)
        ))

# Data Overview page
elif options == "📋 Data Overview":
    st.title("Data Overview")
    
    tab1, tab2, tab3 = st.tabs(["Preview", "Information", "Statistics"])
    
    with tab1:
        st.dataframe(df.head(), use_container_width=True)
    
    with tab2:
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.code(buffer.getvalue())
    
    with tab3:
        st.dataframe(df.describe(), use_container_width=True)

# Visualizations page
elif options == "📊 Visualizations":
    st.title("Data Visualizations")
    
    # Prepare numeric data
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    numeric_df = numeric_df.dropna()
    
    tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "Scatter & Regression", "Category Analysis"])
    
    with tab1:
        st.subheader("Correlation Heatmap")
        fig = px.imshow(
            numeric_df.corr(),
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            scatter_x = st.selectbox("X-axis:", numeric_df.columns, key='scatter_x')
            scatter_y = st.selectbox("Y-axis:", numeric_df.columns, key='scatter_y')
            
            correlation, p_value = pearsonr(numeric_df[scatter_x], numeric_df[scatter_y])
            
            fig = px.scatter(
                numeric_df,
                x=scatter_x,
                y=scatter_y,
                trendline="ols",
                title=f"Correlation: {correlation:.2f} (p-value: {p_value:.2e})"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        categorical_columns = df.select_dtypes(include=["object"]).columns
        cat_x = st.selectbox("Category:", categorical_columns)
        num_y = st.selectbox("Value:", numeric_df.columns)
        
        fig = px.box(
            df,
            x=cat_x,
            y=num_y,
            title=f"Distribution of {num_y} by {cat_x}"
        )
        st.plotly_chart(fig, use_container_width=True)

# Statistics page
elif options == "📈 Statistics":
    st.title("Statistical Analysis")
    
    tab1, tab2 = st.tabs(["Group Analysis", "Distribution Analysis"])
    
    with tab1:
        group_col = st.selectbox("Group by:", df.columns)
        if group_col:
            grouped_stats = df.groupby(group_col).agg(['mean', 'count', 'std']).round(2)
            st.dataframe(grouped_stats, use_container_width=True)
    
    with tab2:
        dist_col = st.selectbox("Select column:", df.columns)
        if df[dist_col].dtype in ['int64', 'float64']:
            fig = px.histogram(
                df,
                x=dist_col,
                title=f"Distribution of {dist_col}",
                marginal="box"
            )
        else:
            value_counts = df[dist_col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {dist_col}"
            )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
    ---
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Created with ❤️ using Streamlit | Data Analysis Dashboard v1.0
    </div>
""", unsafe_allow_html=True)
