# -*- coding: utf-8 -*-

import logging
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import FastTextKeyedVectors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PoetryEmbedder:
    """
    A comprehensive class for training, loading, and using FastText embeddings
    tailored for Arabic poetry. This class handles the end-to-end embedding process,
    from training on a corpus to generating document-level vectors.
    """

    def __init__(self, model: Optional[FastTextKeyedVectors] = None):
        """
        Initializes the PoetryEmbedder.

        Args:
            model (Optional[FastTextKeyedVectors]): A pre-loaded Gensim FastText model.
                                                    If None, a model must be loaded or trained.
        """
        self.model = model
        if self.model:
            self.vector_size = self.model.vector_size
            logging.info(f"Embedder initialized with a pre-loaded model of vector size {self.vector_size}.")
        else:
            self.vector_size = None
            logging.info("Embedder initialized without a model. Load or train a model before use.")

    def train_model(
        self,
        corpus_path: str,
        text_column: str,
        save_path: str,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 2,
        workers: int = -1,
        epochs: int = 10
    ):
        """
        Trains a new FastText model from a text corpus (CSV file).

        Args:
            corpus_path (str): Path to the training corpus CSV file.
            text_column (str): The column name in the CSV containing the preprocessed text.
            save_path (str): Path to save the trained model (.bin file).
            vector_size (int): Dimensionality of the word vectors.
            window (int): The maximum distance between the current and predicted word within a sentence.
            min_count (int): Ignores all words with a total frequency lower than this.
            workers (int): Use these many worker threads to train the model. -1 means use all available cores.
            epochs (int): Number of iterations (epochs) over the corpus.
        """
        logging.info(f"Starting FastText model training from corpus: {corpus_path}")

        try:
            df = pd.read_csv(corpus_path)
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in {corpus_path}.")
            
            # Gensim's FastText expects a list of lists of tokens.
            sentences = df[text_column].dropna().str.split().tolist()
            logging.info(f"Training on {len(sentences)} sentences/documents.")
            
        except FileNotFoundError:
            logging.error(f"Corpus file not found at {corpus_path}.")
            raise
        except Exception as e:
            logging.error(f"Failed to read or process corpus file: {e}")
            raise

        if not sentences:
            logging.error("The processed corpus is empty. Cannot train the model.")
            return

        # Determine the number of workers
        if workers == -1:
            try:
                workers = os.cpu_count() or 4
                logging.info(f"Using {workers} CPU cores for training.")
            except NotImplementedError:
                workers = 4 # Fallback
                logging.warning("Could not determine CPU count. Defaulting to 4 workers.")

        # Train the model
        logging.info("Training FastText model...")
        model = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=1  # Use skip-gram
        )
        logging.info("Model training completed.")

        # Save the model
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            logging.info(f"Model successfully saved to {save_path}")
            
            # Load the trained model into the instance
            self.load_model(save_path)

        except Exception as e:
            logging.error(f"Failed to save the model to {save_path}: {e}")
            raise

    def load_model(self, model_path: str):
        """
        Loads a pre-trained FastText model from a .bin file.

        Args:
            model_path (str): Path to the .bin model file.
        """
        logging.info(f"Loading FastText model from {model_path}...")
        try:
            # We load only the KeyedVectors to save memory if we don't need to continue training.
            self.model = FastText.load(model_path).wv
            self.vector_size = self.model.vector_size
            logging.info(f"Model loaded successfully. Vector size: {self.vector_size}.")
        except FileNotFoundError:
            logging.error(f"Model file not found at {model_path}.")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {e}")
            raise

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Retrieves the vector for a single word. Returns None if the word is not in vocabulary.

        Args:
            word (str): The word to look up.

        Returns:
            Optional[np.ndarray]: The word vector, or None if the model is not loaded.
        """
        if not self.model:
            logging.error("Model is not loaded. Cannot get word vector.")
            return None
        
        if word in self.model:
            return self.model[word]
        else:
            # FastText can generate vectors for out-of-vocabulary words
            # by using its subword information.
            logging.debug(f"Word '{word}' is OOV, generating vector from subwords.")
            return self.model[word]

    def get_document_vector(self, text: Union[str, List[str]], strategy: str = 'mean') -> Optional[np.ndarray]:
        """
        Generates a single vector representation for a document (a string or a list of words).

        Args:
            text (Union[str, List[str]]): The input text. Can be a single string or a list of pre-tokenized words.
            strategy (str): The strategy to aggregate word vectors. Currently only 'mean' is supported.

        Returns:
            Optional[np.ndarray]: A single vector for the document, or None if no valid words are found or model is not loaded.
        """
        if not self.model:
            logging.error("Model is not loaded. Cannot generate document vector.")
            return None

        if isinstance(text, str):
            tokens = text.split()
        elif isinstance(text, list):
            tokens = text
        else:
            logging.error(f"Input text must be a string or a list of strings, but got {type(text)}.")
            return None

        if not tokens:
            logging.warning("Input text is empty. Returning a zero vector.")
            return np.zeros(self.vector_size)

        word_vectors = []
        for token in tokens:
            # We don't check for OOV because FastText handles it.
            word_vectors.append(self.model[token])

        if not word_vectors:
            logging.warning(f"No valid word vectors found for the input text: '{' '.join(tokens)[:50]}...'. Returning a zero vector.")
            return np.zeros(self.vector_size)

        if strategy == 'mean':
            return np.mean(word_vectors, axis=0)
        else:
            logging.error(f"Unsupported aggregation strategy: '{strategy}'. Only 'mean' is available.")
            raise ValueError(f"Unsupported aggregation strategy: '{strategy}'")

    def embed_dataframe_column(self, df: pd.DataFrame, text_column: str, new_column_name: str = 'doc_vector') -> pd.DataFrame:
        """
        Applies document embedding to a DataFrame column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The column containing the preprocessed text.
            new_column_name (str): Name for the new column containing document vectors.

        Returns:
            pd.DataFrame: The DataFrame with the new vector column.
        """
        if not self.model:
            logging.error("Model is not loaded. Cannot embed DataFrame.")
            raise RuntimeError("Model must be loaded before embedding a DataFrame.")
            
        if text_column not in df.columns:
            logging.error(f"Column '{text_column}' not found in the DataFrame.")
            raise ValueError(f"Column '{text_column}' not found.")

        logging.info(f"Generating document vectors for column '{text_column}'...")
        df_copy = df.copy()
        df_copy[new_column_name] = df_copy[text_column].apply(
            lambda x: self.get_document_vector(x) if pd.notna(x) else np.zeros(self.vector_size)
        )
        logging.info("Finished generating document vectors.")
        return df_copy