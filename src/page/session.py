import pandas as pd
import streamlit as st
from copy import deepcopy
    
def init_session(df: pd.DataFrame):
    if "history" not in st.session_state:
        st.session_state.history = []
    if "data" not in st.session_state:
        st.session_state.data = df.copy()

def get_df():
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("Upload your data first.")
        st.stop()

    return st.session_state.data

def confirm(new_df: pd.DataFrame, action_desc: str = ""):
    st.session_state.history.append(
        (deepcopy(st.session_state.data), action_desc)
    )
    st.session_state.data = new_df.copy()

def undo():
    if st.session_state.history:
        last_df, _ = st.session_state.history.pop()
        st.session_state.data = last_df
    else:
        st.warning("No more steps to undo.")