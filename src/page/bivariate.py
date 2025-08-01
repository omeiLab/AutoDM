import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

    eda = BivariateAnalyser(data)
    show_relationship(eda, col1, col2)

@st.cache_data
def preview(data):
    st.write("## Data Preview")      
    st.dataframe(data.head(5))

def enable_hue(eda, f1, f2):
    use_hue = st.checkbox("Enable hue (label coloring)")
    label_col = None
    if use_hue:
        candidates = [c for c in eda.df.columns if eda.dtypes[c] == "categorical" and c not in [f1, f2]]
        if candidates:
            label_col = st.selectbox("Select label column for hue", candidates)
            return label_col
        else:
            st.warning("No valid categorical columns available for hue.")
            return None
    return None

def show_relationship(eda, f1, f2):
    '''
    Analysis status:
        - 1: numerical vs numerical,     summary = correlation coefficient
        - 2: numerical vs categorical,   summary = groupby analysis
        - 3: categorical vs categorical, summary = chi-square test

        - -1: high cardinality,          summary = None
    '''
    if f1 == f2:
        st.warning("Cannot compare a feature with itself.")
        st.stop()
    
    status, plot, summary = eda.analyse(f1, f2)

    if status == -1:
        st.warning(f"High Cardinality. Cannot perform analysis.")
        st.stop()

    elif status == 1:
        label_col = enable_hue(eda, f1, f2)
        if label_col is not None:
            _, plot, summary = eda.analyse(f1, f2, label_col)

        st.markdown(f"""
        ```
        Correlation Coefficient: {summary:.4f}
        ```
        """)
        st.pyplot(plot)
        plt.close(plot)

    elif status == 2:
        label_col = enable_hue(eda, f1, f2)
        if label_col is not None:
            _, plot, summary = eda.analyse(f1, f2, label_col)

        with st.expander("Groupby Analysis"):
            st.dataframe(summary)
        st.pyplot(plot)
        plt.close(plot)

    elif status == 3:
        # print(summary)
        st.markdown(f"""
        ```
        Chi-Square Test: {summary[0]:.4f}
        p-value: {summary[1]:.4f}
        Degree of Freedom: {summary[2]}
        ```
        """)
        if summary[1] < 0.05:
            st.info("The relationship between the two features is significant.")
        st.pyplot(plot)
        plt.close(plot)