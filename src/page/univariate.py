import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from module.EDAnalyser.AnalyserFactory import AnalyserFactory

def page_univariate_eda():
    '''
    Perform univariate EDA on uploaded data.
    '''
    st.title("Univariate EDA")
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("Upload your data first.")
        st.stop()

    data = st.session_state["data"]
    preview(data)
    data_summary(data)

@st.cache_data
def preview(data):
    st.write("## Data Preview")      
    st.dataframe(data.head(5))

def data_summary(data):
    st.write("## Data Summary")  
    overview = summarize_overview(data)

    # Use st.metrics to perform 
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{overview['rows']}")
    col2.metric("Columns", f"{overview['columns']}")
    col3.metric("Missing Values", f"{overview['missing_values']}")

    # Use st.tabs to show details of columns
    cols = data.columns.tolist()
    tabs = st.tabs(cols)
    for col_name, tab in zip(cols, tabs):
        eda = AnalyserFactory.create(data, col_name)
        with tab:
            analyse = eda.analyse(overview) # type: ignore
            dtype = analyse["dtype"]
            st.markdown(f"""
                **Column Information**
                        
                    - 📂 Data Type: {dtype}
                    - ❓ Missing Ratio: {(analyse["missing_ratio"]):.1%}
                """)
            
            ## Datetime should be treated specially due to granularity matters
            if dtype == "datetime":
                period = eda.granularity # type: ignore
                period_tabs = st.tabs(period)
                for p, period_tab in zip(period, period_tabs):
                    with period_tab:
                        plot = eda.visualize_by_period(p) # type: ignore
                        st.pyplot(plot) # type: ignore
                        plt.close(plot) # type: ignore
            else:
                if analyse["summary"] is not None:
                    st.dataframe(analyse["summary"])
                if analyse["plot"] is not None:
                    st.pyplot(analyse["plot"])
                    plt.close(analyse["plot"])


def summarize_overview(df) -> dict:
    '''
    Summarize the overview of the DataFrame.
    '''
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().sum(),
        "missing_by_column": df.isnull().sum().to_dict()
    }