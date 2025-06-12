# -*- coding: utf-8 -*-

import logging
import time

import pandas as pd
import streamlit as st

# Assuming your predictor is in the core directory relative to the web_app folder
from core.predictor import PoetryPredictor

# Configure logger for this page
logger = logging.getLogger(__name__)

# --- Page-Specific Configuration ---
st.set_page_config(
    page_title="Analyze Poem - Arabic Poetry Classifier",
    page_icon="✍️",
    layout="wide",
)

# --- Helper Functions ---

@st.cache_resource
def get_predictor():
    """
    Factory function to load and cache the PoetryPredictor instance.
    This ensures the model is loaded only once across all pages of the app.
    """
    try:
        return PoetryPredictor()
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize the model predictor. Please check logs. Error: {e}")
        logger.error(f"Failed to load predictor: {e}", exc_info=True)
        return None

def display_results(results: list):
    """
    Renders the prediction results in a structured and user-friendly format.
    """
    if not results:
        st.warning("The model did not predict any themes above the confidence threshold for the given text.")
        return

    st.success("Analysis Complete!")
    
    # Create a DataFrame for better visualization
    # Convert probability to a more readable format
    df_data = [
        {"Theme": r["theme"], "Confidence": f"{r['probability']:.2%}"}
        for r in results
    ]
    df = pd.DataFrame(df_data)

    st.markdown("#### Predicted Themes:")
    
    # Use columns for a cleaner layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        top_theme = results[0]["theme"]
        top_prob = results[0]["probability"]
        st.metric(label="Top Predicted Theme", value=top_theme, delta=f"{top_prob:.2%}")
        st.info("Confidence indicates the model's certainty for each predicted theme.")

# --- Main Page UI and Logic ---

def main():
    """Main function to render the 'Analyze Poem' page."""
    
    st.title("✍️ Analyze a New Poem")
    st.markdown("Enter the verses of an Arabic poem below to classify its thematic content. "
                "For best results, please place each verse on a new line.")

    # --- Load Predictor ---
    predictor = get_predictor()
    if not predictor:
        st.stop() # Stop execution of the page if the predictor failed to load

    # --- UI Components for Analysis ---
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # --- Poem Input Area ---
        placeholder_poem = (
            "عَلَى قَدْرِ أَهْلِ العَزْمِ تَأْتِي العَزَائِمُ\n"
            "وَتَأْتِي عَلَى قَدْرِ الكِرَامِ المَكَارِمُ\n"
            "وَتَعْظُمُ فِي عَيْنِ الصَّغِيرِ صِغَارُهَا\n"
            "وَتَصْغُرُ فِي عَيْنِ العَظِيمِ العَظَائِمُ"
        )
        poem_input = st.text_area(
            label="**Enter Poem Text Here:**",
            value=placeholder_poem,
            height=250,
            placeholder="Paste or type the poem here..."
        )
        
    with col2:
        # --- Analysis Options ---
        st.markdown("**Analysis Options**")
        top_k = st.slider(
            label="Number of top themes to display:",
            min_value=1,
            max_value=10,
            value=5, # Default value
            help="Adjust this slider to control how many of the most confident themes are shown."
        )

        # The main action button
        analyze_button = st.button("Analyze Poem", type="primary", use_container_width=True)

    st.markdown("---")
    
    # --- Analysis Execution and Output ---
    if analyze_button:
        # Validate input
        if not poem_input or not poem_input.strip():
            st.warning("⚠️ Input is empty. Please enter a poem to analyze.")
            return # Stop execution for this run

        # Show a spinner while processing
        with st.spinner("🔍 Analyzing the poetic essence... Please wait."):
            try:
                start_time = time.time()
                
                # Call the predictor
                results = predictor.predict(poem_text=poem_input, top_k=top_k)
                
                end_time = time.time()
                logger.info(f"Prediction for input text completed in {end_time - start_time:.2f} seconds.")

                # Display the results
                display_results(results)

            except (ValueError, TypeError) as e:
                # Handle specific, expected errors from the predictor
                st.error(f"❌ Input Error: {e}")
                logger.warning(f"User input error: {e}")
            except Exception as e:
                # Handle unexpected errors
                st.error("An unexpected error occurred during analysis. Please try again or check the logs.")
                logger.error(f"Unexpected prediction error: {e}", exc_info=True)


if __name__ == "__main__":
    main()