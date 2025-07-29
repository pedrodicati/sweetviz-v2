"""
Basic integration tests for sweetviz functionality
"""
import pandas as pd
import pytest
import sweetviz as sv
import numpy as np


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing"""
    np.random.seed(42)
    data = {
        'numeric_col': np.random.normal(0, 1, 100),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
        'boolean_col': np.random.choice([True, False], 100),
        'text_col': [f'text_{i}' for i in range(100)],
        'missing_col': [1, 2, None, 4, None] * 20
    }
    return pd.DataFrame(data)


class TestBasicFunctionality:
    """Test basic sweetviz functionality"""
    
    def test_import_sweetviz(self):
        """Test that sweetviz can be imported"""
        assert hasattr(sv, 'analyze')
        assert hasattr(sv, 'compare')
        assert hasattr(sv, 'compare_intra')
    
    def test_analyze_basic(self, sample_dataframe):
        """Test basic analyze functionality"""
        report = sv.analyze(sample_dataframe)
        assert report is not None
        assert hasattr(report, 'show_html')
    
    def test_compare_basic(self, sample_dataframe):
        """Test basic compare functionality"""
        df1 = sample_dataframe.iloc[:50]
        df2 = sample_dataframe.iloc[50:]
        report = sv.compare([df1, "First Half"], [df2, "Second Half"])
        assert report is not None
        assert hasattr(report, 'show_html')
    
    def test_feature_config(self):
        """Test FeatureConfig functionality"""
        fc = sv.FeatureConfig()
        assert fc is not None
        
    def test_version_info(self):
        """Test that version information is available"""
        assert hasattr(sv, '__version__')
        assert isinstance(sv.__version__, str)


class TestCompatibility:
    """Test compatibility with latest pandas/numpy versions"""
    
    def test_pandas_compatibility(self, sample_dataframe):
        """Test that sweetviz works with current pandas version"""
        assert pd.__version__.startswith('2.')
        # Basic operations should work
        report = sv.analyze(sample_dataframe)
        assert report is not None
        
    def test_numpy_compatibility(self, sample_dataframe):
        """Test that sweetviz works with current numpy version"""
        assert np.__version__.startswith('2.')
        # Numeric operations should work
        numeric_data = pd.DataFrame({'nums': np.random.normal(0, 1, 50)})
        report = sv.analyze(numeric_data)
        assert report is not None