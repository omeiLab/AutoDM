import streamlit as st
from module.data_loader import page_upload_data

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

page_names_to_funcs = {
    "-": page_intro,
    "Upload Data": page_upload_data
}

demo_name = st.sidebar.selectbox("Function Pages", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()