import streamlit as st
import pandas as pd
from module.bivariate_analyser import BivariateAnalyser

def page_bivariate_eda():
    st.title("Bivariate EDA")
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("Upload your data first.")
        st.stop()

    data = st.session_state["data"]
    preview(data)
    
    # choose columns
    st.write("## Data Relationships Analysis")
    st.write("Select two features to compare their relationships.")
    f1, f2 = st.columns(2)
    with f1:
        col1 = st.selectbox("📌 Feature 1", data.columns, key="bivariate_col1")
    with f2:
        col2 = st.selectbox("📌 Feature 2", data.columns, key="bivariate_col2")

    show_relationship(data, col1, col2)

def preview(data):
    st.write("## Data Preview")      
    st.dataframe(data.head(5))

def show_relationship(data, f1, f2):
    '''
    Analysis status:
        - 1: numerical vs numerical,        summary = correlation coefficient
        - 2: numerical vs categorical,      summary = groupby analysis
        - 3: categorical vs categorical,    
    '''
    if f1 == f2:
        st.warning("Cannot compare a feature with itself.")
        st.stop()
    eda = BivariateAnalyser(data)
    status, plot, summary = eda.analyse(f1, f2)

    if status == 1:
        st.markdown(f"""
        ```
        Correlation Coefficient: {summary:.4f}
        ```
        """)
        st.pyplot(plot)
    elif status == 2:
        with st.expander("Groupby Analysis"):
            st.dataframe(summary)
        st.pyplot(plot)
    elif status == 3:
        st.info("Not implemented yet.")