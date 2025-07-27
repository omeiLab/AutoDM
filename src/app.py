import streamlit as st
from module.data_loader import DataLoader
from page.univariate import page_univariate_eda
from page.bivariate import page_bivariate_eda

def page_intro():

    st.write("# AutoDM - Automated Data Mining")
    st.sidebar.success("Select a demo above.")
    st.markdown(
        """
        AutoDM is an automated data mining tool that helps you extract valuable insights from your data.

        You can use AutoDM to:

        - Extract insights from your data
        - Analyze your data for patterns and trends
        - Identify and remove noise from your data
        - Generate reports and visualizations to share your findings with others

        AutoDM is designed to work with tabular data for now. We plan to add support for other data types in the future.
        Click the **Function Pages** button on the sidebar and navigate to **Upload Data** to get started.
        """
    )
    upload_data()

def upload_data():
    '''
    Upload data page for the app.
    '''
    st.write("### Upload Data")
    uploaded_file = st.file_uploader(
        "Please upload your data here. We accept tabular **CSV files** only.", 
        accept_multiple_files=False,
        type = ['csv']
    )
    data = preview(uploaded_file)
    st.session_state["data"] = data

def preview(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            try:
                dl = DataLoader()
                data = dl.load_data(uploaded_file)
                st.success("Data uploaded successfully!")
                st.write("### Data Preview")      
                st.dataframe(data.head(5))
                return data
            except:
                st.error("Error loading data. Please check your file and try again.")
        else:
            st.warning("Invalide file type. Please upload a CSV file.")
            return None

page_names_to_funcs = {
    "-": page_intro,
    "Univariate EDA": page_univariate_eda,
    "Bivariate EDA": page_bivariate_eda,
}

demo_name = st.sidebar.selectbox("Function Pages", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()