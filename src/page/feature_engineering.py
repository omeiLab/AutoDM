import streamlit as st
from streamlit_sortables import sort_items
import pandas as pd
import matplotlib.pyplot as plt
from page.session import get_df, confirm, undo
from module.FeatureProcessingHandler.MissingValuesHandler import MissingValuesHandler
from module.FeatureProcessingHandler.CategoricalEncodingHandler import CategoricalEncodingHandler
from module.FeatureProcessingHandler.NumericalHandler import NumericalHandler

def page_feature_engineering():
    '''
    Perform univariate EDA on uploaded data.
    '''
    st.title("Feature Engineering")
    data = get_df()
    st.write("Here, you can perform feature engineering tasks on the uploaded data.")
    st.write("If you want to undo any changes, click the 'Undo' button below.")
    if st.button("Undo", use_container_width=True):
        undo()
        st.rerun()
    if data.isnull().values.any():
        missing_values_handler(data)
    categorical_encoding_handler(data)
    numerical_handler(data)
    

def missing_values_handler(data):
    st.markdown("## Missing Values")
    st.write("Let's check for missing values in the data. Choose a column to view the missing values and decide how to handle them.")
    missing_values_handler = MissingValuesHandler(data)
    target_col, method = build_missing_current_status(missing_values_handler)
    if target_col != 'None':
        if st.button("Confirm Imputation", use_container_width=True):
            new_df = missing_values_handler.process(target_col, method)
            confirm(new_df, f"Impute {target_col} with {method}")
            st.rerun()

# This is for show the current status of missing values in real-time
def build_missing_current_status(missing_values_handler: MissingValuesHandler):

    # Draw the missing values plot
    expand = st.expander("Missing Values Plot")
    with expand:
        plot = missing_values_handler.plot_missing_values()
        st.pyplot(plot)
        plt.close(plot)

    # column & method
    sug = missing_values_handler.suggest_imputation()
    cols = list(sug.keys())
    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox('Select column', ['None'] + cols, key='selected_missing_col')
    with col2:
        if selected_col == 'None':
            methods = ['Select a column first']
        else:
            methods = ['None'] + missing_values_handler.show_methods(selected_col)
        selected_method = st.selectbox('Select method', methods, key='selected_missing_method')

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
    return selected_col, selected_method

def categorical_encoding_handler(data):
    st.markdown("## Categorical Encoding")
    st.write("Let's encode categorical variables. Choose a categorical column and decide how to encode them.")
    categorical_encoding_handler = CategoricalEncodingHandler(data)
    selected_col, selected_method, mapping = build_encoding_current_status(categorical_encoding_handler, data)
    # buttons to confirm or undo
    if selected_col != 'None':
        if st.button("Confirm Encoding", use_container_width=True):
            new_df = categorical_encoding_handler.process(selected_col, selected_method, mapping)
            confirm(new_df, f"Encode {selected_col} with {selected_method}")
            st.rerun()

def ordinal_sort_ui(colname, categories):
    st.write(f"Define order for: **{colname}**")
    ordered = sort_items(categories, direction="horizontal")
    mapping = {cat: i+1 for i, cat in enumerate(ordered)}
    st.write("📊 Current mapping:", mapping)
    return mapping

def build_encoding_current_status(categorical_encoding_handler: CategoricalEncodingHandler, data):
    # column & method
    sug = categorical_encoding_handler.suggest_encoding()
    cols = list(sug.keys())
    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox('Select column', ['None'] + cols, key='selected_encoding_col')
    with col2:
        if selected_col == 'None':
            methods = ['Select a column first']
        else:
            methods = ['None'] + categorical_encoding_handler.show_methods()
        selected_method = st.selectbox('Select method', methods, key='selected_encoding_method')

    if selected_col != 'None':
        st.info(f"Suggestion: {sug[selected_col]} if your data is nominal. Otherwise, choose label encoding (ordinal) if the cardinality is low.")

    # if method = label encoding (ordinal), show mapping options
    mapping = None
    if selected_method == 'label encoding (ordinal)':
        category_lst = data[selected_col].unique()
        category_lst = [str(cat) for cat in category_lst]
        mapping = ordinal_sort_ui(selected_col, category_lst)
    
    # encoding preview
    if selected_col != 'None' and selected_method != 'None':
        col1, col2 = st.columns(2)
        before, after = categorical_encoding_handler.preview_encoding(selected_col, selected_method, mapping)
        with col1:
            st.dataframe(before)
        with col2:
            st.dataframe(after)
        
    return selected_col, selected_method, mapping

def numerical_handler(data):
    st.markdown("## Numerical Features")
    st.write("Let's perform some numerical feature engineering tasks. Choose a numerical column and decide what to perform.")
    numerical_handler = NumericalHandler(data)
    selected_col, scale_method, outlier_method, transform_method, poly_method = build_numerical_current_status(numerical_handler)
    if selected_col != 'None':
        if st.button("Confirm Processing", use_container_width=True):
            new_df = numerical_handler.process(selected_col, scale_method, outlier_method, transform_method, poly_method)
            confirm(new_df, f"Process {selected_col} with {scale_method}, {outlier_method}, {transform_method}, {poly_method}")
            st.rerun()

def build_numerical_current_status(numerical_handler: NumericalHandler):
    # column & method
    cols = numerical_handler.get_num_cols()
    selected_col = st.selectbox('Select column', ['None'] + cols, key='selected_numerical_col')
    scale_method, outlier_method, transform_method, poly_method = None, None, None, None
    if selected_col != 'None':
        stat_df = numerical_handler.summary(selected_col)
        st.dataframe(stat_df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            scale_method = st.selectbox('Scaling', numerical_handler.get_scale_method(), key='selected_scale_method')
        with col2:
            outlier_method = st.selectbox('Outlier', numerical_handler.get_outlier_method(), key='selected_outlier_method')
        with col3:
            transform_method = st.selectbox('Transform', numerical_handler.get_transform_method(), key='selected_transform_method')
        with col4:
            poly_method = st.selectbox('Polynomial', numerical_handler.get_poly_method(), key='selected_poly_method')

        # preview
        if scale_method != 'None' or outlier_method != 'None' or transform_method != 'None' or poly_method != 'Linear':
            preview_plot = numerical_handler.preview_plot(selected_col, scale_method, outlier_method, transform_method, poly_method)
            st.pyplot(preview_plot)
            plt.close(preview_plot)

    return selected_col, scale_method, outlier_method, transform_method, poly_method
    
