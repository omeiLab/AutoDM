import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from module.EDAnalyser.Bivariate.CatTimeAnalyser import CatTimeAnalyser
from module.utils import classify_dtype
from module.EDAnalyser.AnalyserFactory import BivariateAnalyserFactory

def page_bivariate_eda():
    st.title("Bivariate EDA")
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("Upload your data first.")
        st.stop()

    data = st.session_state["data"]
    preview(data)

    # correlation matrix
    expand = st.expander("Correlation Matrix")
    with expand:
        corr_matrix = correlation_matrix(data)
        st.pyplot(corr_matrix)
        plt.close(corr_matrix)
    
    # choose columns
    st.write("## Data Relationships Analysis")
    st.write("Select two features to compare their relationships.")
    st.write("The hue (label coloring) option is available for categorical features only.")

    f1, f2, h = st.columns(3)
    with f1:
        col1 = st.selectbox("📌 Feature 1", data.columns, key="bivariate_col1")
    with f2:
        col2 = st.selectbox("📌 Feature 2", data.columns, key="bivariate_col2")
    with h:
        candidates = [c for c in data.columns if classify_dtype(data[c]) == "categorical"]
        hue = st.selectbox("🌈 Hue", [None] + candidates, key="bivariate_hue", placeholder=None)

    show_relationship(col1, col2, hue)

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

def show_relationship(f1, f2, hue):
    if f1 == f2:
        st.warning("Cannot compare a feature with itself.")
        st.stop()
    if f1 == hue or f2 == hue:
        st.warning("Cannot enable label coloring while comparing with label column.")
        st.stop()

    eda = BivariateAnalyserFactory.create(st.session_state["data"], f1, f2, hue)
    analyse = eda.analyse() # type: ignore

    # specialize for categorical vs datetime
    if isinstance(eda, CatTimeAnalyser):
        period = eda.granularity 
        period_tabs = st.tabs(period)
        for p, period_tab in zip(period, period_tabs):
            with period_tab:
                analyse = eda.analyse_by_period(p)
                if analyse["summary"] is not None:
                    expand = st.expander(f"{p} Summary")
                    with expand:
                        st.dataframe(analyse["summary"])
                if analyse["plot"] is not None:
                    st.pyplot(analyse["plot"])
                    plt.close(analyse["plot"])
    else:
        if analyse["summary"] is not None:
            expand = st.expander(analyse["name"])
            with expand:
                st.dataframe(analyse["summary"])
        if analyse["plot"] is not None:
            st.pyplot(analyse["plot"])
            plt.close(analyse["plot"])

def correlation_matrix(data):
    numerical_cols = [c for c in data.columns if classify_dtype(data[c]) == "numerical"]
    corr_matrix = data[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    return plt.gcf()