import streamlit as st
import pandas as pd
from module.eda_analyser import EdaAnalyser

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


def preview(data):
    st.write("## Data Preview")      
    st.dataframe(data.head(5))

def data_summary(data):
    if data is not None:
        st.write("## Data Summary")  
        eda = EdaAnalyser(data)
        overview = eda.summarize_overview()

        # Use st.metrics to perform 
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{overview['rows']}")
        col2.metric("Columns", f"{overview['columns']}")
        col3.metric("Missing Values", f"{overview['missing_values']}")

        # Use st.tabs to show details of columns
        cols = data.columns.tolist()
        tabs = st.tabs(cols)
        for col_name, tab in zip(cols, tabs):
            with tab:
                summary = eda.describe_numeric(col_name)
                st.markdown(f"""
                    **Column Information**

                        - 📂 Data Type: {summary["dtype"][0]}
                        - ❓ Missing Ratio: {(overview['missing_by_column'][col_name]/overview['rows']):.1%}
                    """)
                if summary["dtype"][0] == "numerical":
                    descriptive = pd.DataFrame(summary)
                    descriptive = descriptive.drop(['dtype'], axis=1)
                    st.dataframe(descriptive)

                    # visualization
                    plot = eda.plot_numeric(col_name)
                    st.pyplot(plot)
                else:
                    # visualization
                    plot = eda.plot_categorical(col_name)
                    st.pyplot(plot)