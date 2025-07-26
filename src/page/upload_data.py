import streamlit as st
import pandas as pd
from module.data_loader import DataLoader
from module.eda_analyser import EdaAnalyser

def page_upload_data():
    '''
    Upload data page for the app.
    '''
    st.write("# Upload Data")
    uploaded_file = st.file_uploader(
        "Please upload your data here. We accept tabular **CSV files** only.", 
        accept_multiple_files=False,
        type = ['csv']
    )
    data = preview(uploaded_file)

    st.write("## Data Summary")      
    data_summary(data)

def preview(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            try:
                dl = DataLoader()
                data = dl.load_data(uploaded_file)
                st.success("Data uploaded successfully!")
                st.write("## Data Preview")      
                st.dataframe(data.head(5))
                return data
            except:
                st.error("Error loading data. Please check your file and try again.")
        else:
            st.warning("Invalide file type. Please upload a CSV file.")
            return None

def data_summary(data):
    if data is not None:
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