import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from module.FeatureProcessingHandler.MissingValuesHandler import MissingValuesHandler

def page_feature_engineering():
    '''
    Perform univariate EDA on uploaded data.
    '''
    st.title("Feature Engineering")
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("Upload your data first.")
        st.stop()
    
    data = st.session_state["data"]
    missing_values_handler(data)

def missing_values_handler(data):
    st.markdown("## Missing Values")
    st.write("Let's check for missing values in the data.")
    st.write("Choose a column to view the missing values and decide how to handle them.")
    missing_values_handler = MissingValuesHandler(data)
    build_current_status(missing_values_handler)

# This is for show the current status of missing values in real-time
def build_current_status(missing_values_handler: MissingValuesHandler):

    # Draw the missing values plot
    plot = missing_values_handler.plot_missing_values()
    st.pyplot(plot)
    plt.close(plot)

    # column & method
    sug = missing_values_handler.suggest_imputation()
    cols = list(sug.keys())
    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox('Select column', ['None'] + cols, key='selected_col')
    with col2:
        if selected_col == 'None':
            methods = ['Select a column first']
        else:
            methods = ['None'] + missing_values_handler.show_methods(selected_col)
        selected_method = st.selectbox('Select method', methods, key='selected_method')

    if selected_col != 'None':
        st.info(f"Suggestion: {sug[selected_col]}")

    # impute preview
    if selected_col != 'None' and selected_method != 'None':
        
        # show impute preview if method id not "drop column"
        if selected_method != 'drop column' or selected_col != 'drop row':
            impute_preview = missing_values_handler.impute_plot_preview(selected_col, selected_method)
            st.pyplot(impute_preview)
            plt.close(impute_preview)
        else:
            st.info("No preview available for dropping.")
