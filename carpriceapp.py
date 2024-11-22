import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
import io

# Set page config
st.set_page_config(
    page_title="Car Price Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
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
def safe_correlation(x, y):
    try:
        mask = ~(np.isinf(x) | np.isinf(y) | np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 2 or len(y_clean) < 2:
            return 0, 1
        
        return stats.pearsonr(x_clean, y_clean)
    except Exception:
        return 0, 1

# Sidebar
with st.sidebar:
    st.title("Navigation")
    options = st.radio(
        "",
        ["🏠 Home", "📋 Data Overview", "📊 Visualizations", "📈 Statistics"]
    )

# Load data
df = load_data()

if df is None:
    st.error("Failed to load data. Please check the data source and try again.")
    st.stop()

# Home page
if options == "🏠 Home":
    st.title("Car Price Analysis Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            ### 📌 Key Features
            - Interactive car data exploration
            - Dynamic price visualizations
            - Statistical analysis
            - Grouped insights by car features
        """)
    
    with col2:
        st.markdown("""
            ### 🎯 Dataset Overview
            - Total Cars: {}
            - Features: {}
            - Numeric Features: {}
            - Categorical Features: {}
        """.format(
            len(df),
            len(df.columns),
            len(df.select_dtypes(include=['float64', 'int64']).columns),
            len(df.select_dtypes(include=['object']).columns)
        ))

# Data Overview page
elif options == "📋 Data Overview":
    st.title("Car Data Overview")
    
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Data Info", "Statistics"])
    
    with tab1:
        st.dataframe(df.head(), use_container_width=True)
    
    with tab2:
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    with tab3:
        st.dataframe(df.describe(), use_container_width=True)

# Visualizations page
elif options == "📊 Visualizations":
    st.title("Car Data Visualizations")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "Price Relationships", "Category Analysis"])
    
    with tab1:
        if len(numeric_cols) > 1:
            st.subheader("Feature Correlation Heatmap")
            corr_df = df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            
            fig.update_layout(
                height=600,
                width=800,
                title="Feature Correlation Analysis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient numeric columns for correlation analysis")
    
    with tab2:
        if len(numeric_cols) > 1:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select Feature (X axis)", numeric_cols, key='scatter_x')
            with col2:
                y_col = st.selectbox("Select Feature (Y axis)", numeric_cols, key='scatter_y')
            
            if x_col and y_col:
                correlation, p_value = safe_correlation(df[x_col], df[y_col])
                
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"Relationship Analysis (Correlation: {correlation:.2f}, p-value: {p_value:.2e})"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient numeric columns for relationship analysis")
    
    with tab3:
        if categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("Select Category", categorical_cols)
            with col2:
                num_col = st.selectbox("Select Numeric Feature", numeric_cols)
            
            if cat_col and num_col:
                fig = px.box(
                    df,
                    x=cat_col,
                    y=num_col,
                    title=f"{num_col} Distribution by {cat_col}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need both categorical and numeric features for this analysis")

# Statistics page
elif options == "📈 Statistics":
    st.title("Statistical Analysis")
    
    tab1, tab2 = st.tabs(["Group Analysis", "Distribution Analysis"])
    
    with tab1:
        group_col = st.selectbox("Group by:", df.columns)
        if group_col:
            try:
                grouped_stats = df.groupby(group_col).agg(['mean', 'count', 'std']).round(2)
                st.dataframe(grouped_stats, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not perform grouping: {str(e)}")
    
    with tab2:
        dist_col = st.selectbox("Select feature:", df.columns)
        if dist_col:
            try:
                if df[dist_col].dtype in ['int64', 'float64']:
                    fig = px.histogram(
                        df,
                        x=dist_col,
                        title=f"Distribution of {dist_col}"
                    )
                else:
                    value_counts = df[dist_col].value_counts()
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribution of {dist_col}"
                    )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create distribution plot: {str(e)}")

# Footer
st.markdown("""
    ---
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Car Price Analysis Dashboard | Created with Streamlit
    </div>
""", unsafe_allow_html=True)

Version 2 of 2
