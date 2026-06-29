"""Small helpers for safely displaying pandas DataFrames in Streamlit."""

import pandas as pd


def clean_dataframe_for_streamlit(value) -> pd.DataFrame:
    """Return a DataFrame with attrs cleared before passing it to Streamlit.

    Streamlit tries to serialize DataFrame attrs. Pandas attrs are useful for
    internal metadata, but they should not contain complex objects like nested
    DataFrames when displayed.
    """

    if value is None:
        cleaned_dataframe = pd.DataFrame()
    elif isinstance(value, pd.DataFrame):
        cleaned_dataframe = value.copy()
    elif isinstance(value, list):
        cleaned_dataframe = pd.DataFrame(value)
    else:
        cleaned_dataframe = pd.DataFrame(value)

    cleaned_dataframe.attrs = {}

    return cleaned_dataframe
