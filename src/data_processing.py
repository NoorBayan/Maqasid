# -*- coding: utf-8 -*-

import pandas as pd
import re
import string
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from pyfarasa.segmenter import FarasaSegmenter
except ImportError:
    logging.error("pyfarasa is not installed. Please install it using 'pip install pyfarasa'.")
    # You might want to raise the error to stop execution if Farasa is critical
    # raise ImportError("pyfarasa is not installed.")
    FarasaSegmenter = None # Set to None to handle cases where it's not present


class PoetryPreprocessor:
    """
    A robust class to preprocess Arabic poetry text.
    Handles cleaning, normalization, stopword removal, and morphological segmentation.
    Includes error handling and logging.
    """

    def __init__(self, stopwords_path: Optional[str] = None):
        """
        Initializes the preprocessor, setting up the Farasa segmenter and loading stopwords.

        Args:
            stopwords_path (Optional[str]): Path to a file containing custom stopwords, one per line.
                                            If None, a default internal list is used.
        """
        self.segmenter = self._initialize_farasa()
        self.stopwords = self._load_stopwords(stopwords_path)

    def _initialize_farasa(self):
        """
        Initializes the FarasaSegmenter and handles potential errors if pyfarasa is not installed
        or if the JAR files are not found.
        """
        if FarasaSegmenter is None:
            logging.warning("FarasaSegmenter is not available. Segmentation step will be skipped.")
            return None
        try:
            # interactive=True avoids re-initializing the JVM for each call, making it much faster.
            return FarasaSegmenter(interactive=True)
        except Exception as e:
            logging.error(f"Failed to initialize FarasaSegmenter. Error: {e}")
            logging.warning("Segmentation step will be skipped. Ensure Java is installed and Farasa's dependencies are correct.")
            return None

    def _load_stopwords(self, path: Optional[str]) -> List[str]:
        """
        Loads stopwords from a specified file path or returns a default list.
        """
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    stopwords = [line.strip() for line in f if line.strip()]
                logging.info(f"Successfully loaded {len(stopwords)} stopwords from {path}.")
                return stopwords
            except FileNotFoundError:
                logging.error(f"Stopwords file not found at {path}. Falling back to default list.")
        
        # Fallback to default list
        logging.info("Using default internal stopwords list.")
        return [
            "من", "في", "على", "إلى", "عن", "و", "ف", "ثم", "أو", "ب", "ك", "ل",
            "يا", "ما", "لا", "إن", "أن", "كان", "قد", "لقد", "لكن", "هذا", "هذه",
            "ذلك", "تلك", "هنا", "هناك", "هو", "هي", "هم", "هن", "أنا", "نحن",
            "أنت", "أنتي", "أنتما", "أنتن", "الذي", "التي", "الذين", "اللاتي",
            "كل", "بعض", "غير", "سوى", "مثل", "كيف", "متى", "أين", "كم", "أي",
            "حتى", "إذ", "إذا", "إذن", "بعد", "قبل", "حين", "بين", "مع", "عند"
        ]

    def _clean_text(self, text: str) -> str:
        """
        Applies basic cleaning to the text: removes diacritics, non-Arabic chars,
        tatweel, and extra whitespaces.
        Handles non-string inputs gracefully.
        """
        if not isinstance(text, str):
            logging.warning(f"Input is not a string (type: {type(text)}), returning empty string.")
            return ""
        
        # 1. Remove diacritics (tashkeel)
        text = re.sub(r'[\u064B-\u0652]', '', text)
        
        # 2. Remove tatweel (elongation)
        text = re.sub(r'\u0640', '', text)

        # 3. Remove punctuation and non-Arabic characters (keeps only Arabic letters and spaces)
        # This regex is more comprehensive and includes a wider range of punctuation.
        arabic_only_pattern = re.compile(r'[^\u0621-\u064A\s]')
        text = arabic_only_pattern.sub('', text)

        # 4. Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _normalize_arabic(self, text: str) -> str:
        """
        Normalizes Arabic characters: unifies Alef forms, Yaa/Alef Maqsura, and Taa Marbuta.
        """
        text = re.sub(r'[إأآ]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ة', 'ه', text)
        return text

    def _remove_stopwords(self, text: str) -> str:
        """Removes stopwords from a space-tokenized text."""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return " ".join(filtered_words)

    def _segment_text(self, text: str) -> str:
        """

        Segments the text using the initialized Farasa segmenter.
        Returns the original text if the segmenter is not available.
        """
        if self.segmenter and text:
            try:
                # Farasa returns a single string with '+' delimiters.
                segmented_text = self.segmenter.segment(text)
                return segmented_text
            except Exception as e:
                logging.error(f"Error during Farasa segmentation for text: '{text[:50]}...'. Error: {e}")
                return text # Return original text on error
        return text # Return original text if no segmenter or empty text

    def process_text(self, text: str) -> str:
        """

        Applies the full preprocessing pipeline to a single string of text.
        The order of operations is crucial for correctness and efficiency.
        """
        # 1. Clean first to remove noise
        cleaned_text = self._clean_text(text)
        
        # 2. Normalize characters
        normalized_text = self._normalize_arabic(cleaned_text)
        
        # 3. Remove stopwords
        no_stopwords_text = self._remove_stopwords(normalized_text)
        
        # 4. Segment the remaining text using Farasa
        segmented_text = self._segment_text(no_stopwords_text)
        
        return segmented_text

    def process_dataframe(self, df: pd.DataFrame, text_column: str, new_column_name: str = 'processed_text') -> pd.DataFrame:
        """
        Applies the full preprocessing pipeline to a pandas DataFrame column.
        Includes a progress log for large dataframes.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The name of the column containing the text to process.
            new_column_name (str): The name for the new column with processed text.

        Returns:
            pd.DataFrame: DataFrame with the new processed text column.
        """
        if text_column not in df.columns:
            logging.error(f"Column '{text_column}' not found in the DataFrame. Available columns: {df.columns.tolist()}")
            raise ValueError(f"Column '{text_column}' not found.")

        df_copy = df.copy()
        
        logging.info(f"Starting preprocessing for column '{text_column}' in a DataFrame with {len(df_copy)} rows.")
        
        # Using .apply() is convenient. For very large datasets, consider swifter or Dask for parallelization.
        df_copy[new_column_name] = df_copy[text_column].apply(self.process_text)
        
        logging.info("Finished preprocessing the DataFrame.")
        
        # Check for empty results which might indicate an issue.
        empty_results = df_copy[df_copy[new_column_name] == ''].shape[0]
        if empty_results > 0:
            logging.warning(f"{empty_results} rows resulted in an empty string after processing. Please check your data and preprocessing steps.")

        return df_copy