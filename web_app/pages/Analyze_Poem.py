# -*- coding: utf-8 -*-

import json
import logging
import os
import time

import pandas as pd
import plotly.express as px
import streamlit as st

# Path adjustments for importing from the project's src directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from web_app.core.predictor import PoetryPredictor
except ImportError:
    st.error("Could not import the PoetryPredictor. Ensure the project structure is correct.")
    st.stop()


# --- Configure logger ---
logger = logging.getLogger(__name__)


# --- Page Configuration ---
try:
    st.set_page_config(
        page_title="Analyze Poem",
        page_icon="✍️",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass


# --- Caching and Data Loading ---

@st.cache_resource
def get_predictor():
    """Factory function to load and cache the PoetryPredictor instance."""
    try:
        return PoetryPredictor()
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}", exc_info=True)
        return None

@st.cache_data
def load_example_poems(file_path: str = "web_app/static/example_poems.json"):
    """
    Loads example poems from an external JSON file and caches the result.
    This makes the app data-driven and easy to update.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            examples_list = json.load(f)
        # Convert list of dicts to a single dict for easy lookup
        return {item['key']: item['value'] for item in examples_list}
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to load or parse example poems from {file_path}: {e}")
        st.error("Could not load example poems file.")
        return {"Default": "Please add examples to static/example_poems.json"}


# --- UI Helper Functions ---

def render_results(results):
    """Renders the analysis results in a visually appealing and informative way."""
    if not results:
        st.warning("The model did not predict any themes above the confidence threshold.")
        return

    st.success("Analysis Complete!")
    
    df = pd.DataFrame(results)
    df['probability_percent'] = df['probability'].apply(lambda p: f"{p:.2%}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### **Top Predicted Themes**")
        st.dataframe(
            df[['theme', 'probability_percent']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "theme": st.column_config.TextColumn("Theme", width="large"),
                "probability_percent": st.column_config.TextColumn("Confidence"),
            }
        )

    with col2:
        st.markdown("#### **Confidence Distribution**")
        fig = px.bar(
            df, x='probability', y='theme', orientation='h', text='probability_percent',
            labels={'probability': 'Confidence Score', 'theme': 'Thematic Category'},
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        fig.update_traces(marker_color='#0068c9', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)


# --- Main Page Application ---

def main():
    """Main function to render the 'Analyze Poem' page."""
    
    st.title("✍️ Thematic Analysis of a Poem")
    st.markdown("Paste or select an example of Arabic poetry below to classify its thematic content.")
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    
    predictor = get_predictor()
    example_poems = load_example_poems()

    if predictor is None:
        st.error("The analysis model is not available. Application cannot proceed.")
        st.stop()

    # --- Input Section ---
    st.markdown("---")
    
    input_col, example_col = st.columns([2, 1])
    
    with example_col:
        st.markdown("##### **Try an Example**")
        if example_poems:
            selected_example_key = st.selectbox(
                "Choose a famous verse:",
                options=list(example_poems.keys()),
                index=0,
                label_visibility="collapsed"
            )
            # Use the selected key to get the poem text
            default_text = example_poems.get(selected_example_key, "")
        else:
            default_text = "Examples could not be loaded."
            st.warning("No examples available.")

    with input_col:
        st.markdown("##### **Enter Your Poem**")
        poem_input = st.text_area(
            "Paste the poem text here, with each verse on a new line.",
            value=default_text,
            height=200,
            key="poem_input_area"
        )

    # --- Options and Action Button ---
    st.markdown("##### **Analysis Configuration**")
    opts_col1, opts_col2 = st.columns([1, 3])
    with opts_col1:
        top_k = st.number_input(
            "Number of themes to show:", min_value=1, max_value=20, value=7,
            help="The maximum number of most confident themes to display."
        )

    with opts_col2:
        analyze_button = st.button("Analyze Poem", type="primary", use_container_width=True)

    # --- Analysis and Result Display Section ---
    st.markdown("---")
    
    if analyze_button:
        if not poem_input or not poem_input.strip():
            st.warning("⚠️ Please enter a poem to analyze.")
        else:
            with st.spinner("🔬 Performing deep semantic analysis..."):
                try:
                    results = predictor.predict(poem_text=poem_input, top_k=top_k)
                    st.session_state.analysis_results = results
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.analysis_results = None
    
    if st.session_state.analysis_results is not None:
        render_results(st.session_state.analysis_results)

if __name__ == "__main__":
    main()