# -*- coding: utf-8 -*-

import logging
import os
import sys

import streamlit as st

# --- Path Setup ---
# Add the project root to the Python path. This allows importing from 'src' and 'web_app.core'.
# This should be done once in the main entry point of the application.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from web_app.core.predictor import PoetryPredictor
except ImportError:
    # This is a critical failure, the app cannot run without the predictor.
    st.error(
        "**Fatal Error:** Could not import `PoetryPredictor` from `web_app.core`."
        "Please ensure the project structure is correct and all dependencies are installed."
    )
    st.stop()


# --- Application-Wide Configuration ---

# 1. Page Configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Arabic Poetry Thematic Classifier",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-username/your-repo/issues',
        'Report a bug': "https://github.com/your-username/your-repo/issues",
        'About': """
        ## Nuanced Thematic Classification of Arabic Poetry
        This is an interactive platform demonstrating a deep learning model for classifying themes in Arabic poetry.
        Developed as part of our research paper.
        """
    }
)

# 2. Setup application-wide logging
# You can use your utility function here if you have one
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Resource Caching ---

@st.cache_resource
def load_resources():
    """
    Loads all heavy, application-wide resources once and caches them.
    This includes the main model predictor. This function will be called by
    all pages to ensure they share the same cached resources.
    """
    logger.info("Main app: Loading application-wide resources...")
    try:
        predictor = PoetryPredictor()
        logger.info("Main app: PoetryPredictor loaded and cached successfully.")
        return {"predictor": predictor}
    except Exception as e:
        logger.error(f"Main app: A fatal error occurred during resource loading: {e}", exc_info=True)
        # Display the error prominently on the main page
        st.error(
            "**Application Failed to Start**\n\n"
            f"A critical error occurred while loading the AI models: **{e}**\n\n"
            "Please check the application logs for more details. The app will not be functional."
        )
        return {"predictor": None}


# --- Main Application UI ---

def main():
    """
    Renders the main landing page of the application.
    This page introduces the project and guides the user to the sub-pages.
    """

    # --- Load shared resources ---
    # This call ensures that the models are loaded when the user first visits the app.
    # The result is cached, so subsequent calls on this or other pages are instantaneous.
    resources = load_resources()
    if resources["predictor"] is None:
        st.stop() # Halt execution if models failed to load.

    # --- Header and Introduction ---
    st.title("📜 Nuanced Thematic Classification of Arabic Poetry")
    
    st.markdown(
        """
        Welcome to the interactive platform for our research on the thematic analysis of Arabic poetry. 
        This application serves as a practical demonstration of our hybrid deep learning model, 
        designed to understand the rich and complex themes inherent in this literary tradition.
        """
    )
    
    st.info(
        "**👈 To get started, please select a tool from the sidebar on the left.**",
        icon="ℹ️"
    )
    
    # --- Page Navigation Guide ---
    st.header("Available Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("#### ✍️ Analyze a New Poem")
            st.markdown(
                "Input your own Arabic poetry verses or choose from our examples to get an "
                "instant thematic analysis, complete with confidence scores for each predicted theme."
            )
    
    with col2:
        with st.container(border=True):
            st.markdown("#### 📚 Explore the Corpus")
            st.markdown(
                "Dive into our richly annotated dataset. Use interactive filters to explore "
                "thematic trends across different poets, historical eras, and literary genres."
            )
            
    st.markdown("---")

    # --- Project Background and Resources ---
    st.header("About The Project")

    with st.expander("Click here to learn more about the project's methodology and contributions"):
        st.markdown(
            """
            This project was born out of the need to address the significant challenges in the computational 
            analysis of Arabic poetry, namely its thematic complexity, frequent thematic overlap, and the 
            scarcity of comprehensively annotated data.
            
            #### Key Contributions:
            1.  **A Novel Annotated Corpus:** We meticulously created a large-scale, multi-label Arabic poetry corpus, 
                annotated according to a new, literature-grounded hierarchical thematic taxonomy and validated by domain experts.
            
            2.  **An Optimized Hybrid Model:** We developed a hybrid deep learning model that synergistically integrates 
                Convolutional Neural Networks (CNNs) for local feature extraction with Bidirectional Long Short-Term 
                Memory (Bi-LSTM) networks for sequential context modeling. This model leverages custom FastText embeddings 
                trained from scratch on our poetic corpus.
                
            3.  **This Interactive Platform:** This web application makes our model and data accessible, fostering 
                more sophisticated, data-driven research in Arabic digital humanities.
            """
        )
        
    st.markdown("#### Project Links")
    st.link_button("📄 View the Research Paper", "https://your-paper-link.com")
    st.link_button("💻 Browse the Source Code on GitHub", "https://github.com/your-username/your-repo")
    
    # --- Sidebar Content ---
    st.sidebar.success("Select a page above to begin.")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application is a research prototype. "
        "For questions or feedback, please open an issue on our GitHub repository."
    )


if __name__ == "__main__":
    main()