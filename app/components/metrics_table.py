import pandas as pd
import streamlit as st

def render_metrics_table(df: pd.DataFrame, title: str = "Experimental Results Table"):
    """
    Render a clean styled dataframe table in Streamlit.
    """
    st.markdown(f"### {title}")
    st.dataframe(df, use_container_width=True)

def render_csv_download(df: pd.DataFrame, filename: str = "experiment_results.csv"):
    """
    Provide CSV download button for experiment metrics.
    """
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Results as CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv"
    )
