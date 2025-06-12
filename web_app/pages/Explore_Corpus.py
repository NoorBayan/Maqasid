# -*- coding: utf-8 -*-

import logging
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

# Configure logger for this page
logger = logging.getLogger(__name__)


# --- Page Configuration ---
try:
    st.set_page_config(
        page_title="Explore The Corpus",
        page_icon="📚",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    # This error is expected if set_page_config is already called in the main app.py
    pass


# --- Caching and Data Loading ---

@st.cache_data
def load_annotated_corpus(file_path: str = "data/annotated/diwan_corpus_annotated.csv") -> pd.DataFrame:
    """
    Loads the main annotated corpus from a CSV file and caches it.
    This function is the single source of truth for the dashboard's data.
    """
    try:
        logger.info(f"Loading annotated corpus from {file_path}...")
        df = pd.read_csv(file_path)
        # Basic validation
        if 'poet_name' not in df.columns or 'era' not in df.columns:
            raise ValueError("Required columns 'poet_name' or 'era' not found.")
        logger.info(f"Corpus loaded successfully with {len(df)} poems.")
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at '{file_path}'. Please ensure the annotated corpus is available.")
        logger.error(f"Data file not found: {file_path}")
        return pd.DataFrame() # Return empty dataframe to prevent crash
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        logger.error(f"Error loading corpus: {e}", exc_info=True)
        return pd.DataFrame()


@st.cache_data
def get_theme_columns(df: pd.DataFrame) -> List[str]:
    """Extracts theme column names from the DataFrame."""
    return [col for col in df.columns if col.startswith('theme_')]


# --- UI and Plotting Functions ---

def display_main_metrics(df: pd.DataFrame, theme_cols: List[str]):
    """Displays key statistics about the corpus."""
    st.header("Corpus Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Poems", f"{len(df):,}")
    col2.metric("Unique Poets", f"{df['poet_name'].nunique():,}")
    col3.metric("Historical Eras", f"{df['era'].nunique():,}")
    col4.metric("Annotated Themes", f"{len(theme_cols):,}")


def display_theme_distribution(df: pd.DataFrame, theme_cols: List[str]):
    """Calculates and displays the overall distribution of themes."""
    st.subheader("Overall Theme Distribution")
    
    if not theme_cols:
        st.warning("No theme columns found to analyze.")
        return

    # Calculate theme counts
    theme_counts = df[theme_cols].sum().sort_values(ascending=False)
    theme_counts.index = theme_counts.index.str.replace('theme_', '') # Clean up names for display
    
    # Create a bar chart
    fig = px.bar(
        theme_counts,
        x=theme_counts.values,
        y=theme_counts.index,
        orientation='h',
        labels={'x': 'Number of Poems', 'y': 'Theme'},
        title="Frequency of Each Theme in the Corpus"
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def display_interactive_filters(df: pd.DataFrame, theme_cols: List[str]):
    """Creates and manages the interactive filters in the sidebar."""
    st.sidebar.header("🔍 Interactive Filters")
    
    # Theme filter
    # Clean up names for the multiselect widget
    cleaned_theme_names = [name.replace('theme_', '') for name in theme_cols]
    selected_themes_cleaned = st.sidebar.multiselect(
        "Filter by Theme(s):",
        options=cleaned_theme_names,
        help="Select one or more themes to view poems containing them."
    )
    # Map back to original column names
    selected_themes_original = ['theme_' + name for name in selected_themes_cleaned]

    # Poet filter
    all_poets = sorted(df['poet_name'].unique())
    selected_poets = st.sidebar.multiselect(
        "Filter by Poet(s):",
        options=all_poets
    )

    # Era filter
    all_eras = sorted(df['era'].unique())
    selected_eras = st.sidebar.multiselect(
        "Filter by Era(s):",
        options=all_eras
    )

    return selected_themes_original, selected_poets, selected_eras


def apply_filters(df: pd.DataFrame, themes: List[str], poets: List[str], eras: List[str]) -> pd.DataFrame:
    """Applies the selected filters to the DataFrame."""
    filtered_df = df.copy()
    
    # Apply theme filter (poems must contain ALL selected themes)
    if themes:
        for theme in themes:
            filtered_df = filtered_df[filtered_df[theme] == 1]
    
    # Apply poet filter
    if poets:
        filtered_df = filtered_df[filtered_df['poet_name'].isin(poets)]
        
    # Apply era filter
    if eras:
        filtered_df = filtered_df[filtered_df['era'].isin(eras)]
        
    return filtered_df


# --- Main Page Application ---

def main():
    """Main function to render the 'Explore Corpus' page."""
    
    st.title("📚 Explore the Annotated Poetry Corpus")
    st.markdown(
        "This interactive dashboard allows you to explore the thematic landscape of our annotated "
        "Arabic poetry corpus. Use the filters in the sidebar to drill down into the data."
    )
    st.markdown("---")

    # Load data
    corpus_df = load_annotated_corpus()
    
    # Stop if data loading failed
    if corpus_df.empty:
        st.warning("Cannot display dashboard because the data could not be loaded.")
        st.stop()
        
    theme_columns = get_theme_columns(corpus_df)
    
    # Display widgets and get filter selections
    selected_themes, selected_poets, selected_eras = display_interactive_filters(corpus_df, theme_columns)
    
    # Apply filters
    filtered_data = apply_filters(corpus_df, selected_themes, selected_poets, selected_eras)
    
    # --- Main Content Area ---
    
    # Display high-level stats of the *original* corpus
    display_main_metrics(corpus_df, theme_columns)
    st.markdown("---")

    # Display analysis based on the *filtered* data
    st.header("Filtered Results")
    st.write(f"**Showing {len(filtered_data)} of {len(corpus_df)} poems based on your selections.**")
    
    if len(filtered_data) > 0:
        # Display theme distribution for the filtered subset
        display_theme_distribution(filtered_data, theme_columns)
        
        # Display a sample of the filtered data
        st.subheader("Filtered Data Sample")
        st.dataframe(
            filtered_data[['poet_name', 'era', 'poem_text']].head(20),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No poems match the selected filter criteria. Please broaden your search.")


if __name__ == "__main__":
    main()