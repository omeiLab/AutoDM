import streamlit as st
import pandas as pd

def load_data(file) -> pd.DataFrame:
    '''
    Load data from a CSV file.
    '''
    data = pd.read_csv(file)
    return data

def preview(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            try:
                data = load_data(uploaded_file)
                st.success("Data uploaded successfully!")
                st.write("## Data Preview")
                st.dataframe(data.head())
            except:
                st.error("Error loading data. Please check your file and try again.")
        else:
            st.warning("Invalide file type. Please upload a CSV file.")

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
    preview(uploaded_file)
    