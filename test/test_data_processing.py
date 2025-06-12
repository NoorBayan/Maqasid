# -*- coding: utf-8 -*-

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add the source directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from poetry_classifier.data_processing import PoetryPreprocessor

# --- Helper Function to Load Test Data ---

def load_test_data(file_name: str = "processing_examples.json"):
    """Loads test examples from a JSON file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "test_data", file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail(f"Test data file not found at {file_path}. Please create it.")
    except json.JSONDecodeError:
        pytest.fail(f"Could not decode JSON from {file_path}. Please check its syntax.")

# Load all test data once, making it available to all tests
test_data = load_test_data()

# --- Fixtures for Pytest ---

@pytest.fixture(scope="module")
def segmentation_map_from_data():
    """
    Creates the segmentation map for the mock segmenter directly from the test data file.
    This fixture is now the single source of truth for the mock's behavior.
    """
    pipeline_cases = test_data.get("full_pipeline", [])
    if not pipeline_cases:
        pytest.fail("No 'full_pipeline' cases found in test data file.")
    
    return {
        case['expected_pre_segmentation']: case['expected_final']
        for case in pipeline_cases
    }

@pytest.fixture(scope="module")
def mock_farasa_segmenter(segmentation_map_from_data):
    """Mocks the FarasaSegmenter in a purely data-driven way using the segmentation map."""
    mock_segmenter = MagicMock()

    def mock_segment(text_to_segment):
        return segmentation_map_from_data.get(text_to_segment, text_to_segment)

    mock_segmenter.segment.side_effect = mock_segment
    return mock_segmenter


@pytest.fixture
def preprocessor(mock_farasa_segmenter):
    """Provides a PoetryPreprocessor instance with a mocked FarasaSegmenter."""
    with patch('poetry_classifier.data_processing.FarasaSegmenter', return_value=mock_farasa_segmenter):
        yield PoetryPreprocessor()


# --- Test Cases ---

class TestPoetryPreprocessor:

    @pytest.mark.parametrize("case", test_data.get("cleaning", []), ids=[c['name'] for c in test_data.get("cleaning", [])])
    def test_clean_text(self, preprocessor, case):
        """Test the _clean_text method using data-driven cases."""
        if 'input' in case and case['input'] is None:
             assert preprocessor._clean_text(None) == case['expected']
        else:
             assert preprocessor._clean_text(case['input']) == case['expected']

    @pytest.mark.parametrize("case", test_data.get("normalization", []), ids=[c['name'] for c in test_data.get("normalization", [])])
    def test_normalize_arabic(self, preprocessor, case):
        """Test the _normalize_arabic method using data-driven cases."""
        assert preprocessor._normalize_arabic(case['input']) == case['expected']

    @pytest.mark.parametrize("case", test_data.get("stopwords", []), ids=[c['name'] for c in test_data.get("stopwords", [])])
    def test_remove_stopwords(self, preprocessor, case):
        """Test the _remove_stopwords method using data-driven cases."""
        assert preprocessor._remove_stopwords(case['input']) == case['expected']

    @pytest.mark.parametrize("case", test_data.get("full_pipeline", []), ids=[c['name'] for c in test_data.get("full_pipeline", [])])
    def test_process_text_full_pipeline(self, preprocessor, case):
        """Test the end-to-end process_text method using purely data-driven cases."""
        processed_output = preprocessor.process_text(case['input'])
        assert processed_output == case['expected_final']

    def test_process_dataframe(self, preprocessor):
        """Test processing a full pandas DataFrame in a data-driven way."""
        pipeline_cases = test_data.get("full_pipeline", [])
        
        sample_data = {
            'id': [i+1 for i in range(len(pipeline_cases))],
            'raw_verse': [case['input'] for case in pipeline_cases]
        }
        df = pd.DataFrame(sample_data)
        
        processed_df = preprocessor.process_dataframe(df, text_column='raw_verse')
        
        assert 'processed_text' in processed_df.columns
        assert len(processed_df) == len(pipeline_cases)
        for i, case in enumerate(pipeline_cases):
            assert processed_df.loc[i, 'processed_text'] == case['expected_final']

    def test_process_dataframe_with_missing_column(self, preprocessor):
        """Test that processing a DataFrame with a missing column raises a ValueError."""
        df = pd.DataFrame({'id': [1], 'another_col': ['text']})
        
        with pytest.raises(ValueError, match="Column 'non_existent_column' not found"):
            preprocessor.process_dataframe(df, text_column='non_existent_column')