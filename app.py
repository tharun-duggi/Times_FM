"""Streamlit entry point: stock price forecasting with Google's TimesFM 2.5.

Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="TimesFM Stock Forecast", page_icon="📈", layout="wide")

page = st.navigation(
    [
        st.Page("views/single_stock.py", title="Single stock", icon="📈", default=True),
        st.Page("views/document_upload.py", title="Upload document", icon="📄"),
    ]
)
page.run()
