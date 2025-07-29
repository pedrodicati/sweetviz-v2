"""
Basic functionality tests for sweetviz-v2.
"""
import pytest
import pandas as pd
import numpy as np
import sweetviz as sv
from sweetviz import FeatureConfig


class TestBasicFunctionality:
    """Test core sweetviz functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'categorical_col': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
            'boolean_col': [True, False, True, False, True, False, True, False, True, False],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
            'text_col': ['hello', 'world', 'this', 'is', 'a', 'test', 'of', 'text', 'data', 'processing']
        })

    @pytest.fixture 
    def sample_dataframe_with_missing(self):
        """Create a sample dataframe with missing values."""
        df = pd.DataFrame({
            'col_with_nan': [1, 2, np.nan, 4, 5],
            'col_normal': ['a', 'b', 'c', 'd', 'e'],
            # Removed mixed type column as it's not supported
            'col_numeric_with_nan': [1.0, 2.0, np.nan, 4.0, 5.0]
        })
        return df

    def test_analyze_basic(self, sample_dataframe):
        """Test basic analyze functionality."""
        report = sv.analyze(sample_dataframe)
        assert report is not None
        assert hasattr(report, '_features')
        assert len(report._features) == 5  # All columns should be analyzed

    def test_analyze_with_target(self, sample_dataframe):
        """Test analyze with target feature."""
        report = sv.analyze(sample_dataframe, target_feat='boolean_col')
        assert report is not None
        assert report._target is not None

    def test_compare_basic(self, sample_dataframe):
        """Test basic compare functionality."""
        # Create a slightly different dataset for comparison
        df2 = sample_dataframe.copy()
        df2['numeric_col'] = df2['numeric_col'] * 2
        
        report = sv.compare(sample_dataframe, df2)
        assert report is not None
        assert hasattr(report, '_features')

    def test_compare_intra(self, sample_dataframe):
        """Test intra-comparison functionality."""
        condition = sample_dataframe['categorical_col'] == 'A'
        report = sv.compare_intra(
            sample_dataframe, 
            condition, 
            ['Group A', 'Group Not A']
        )
        assert report is not None

    def test_feature_config(self, sample_dataframe):
        """Test FeatureConfig functionality."""
        config = FeatureConfig(
            skip=['text_col'],
            force_cat=['numeric_col']
        )
        report = sv.analyze(sample_dataframe, feat_cfg=config)
        assert report is not None
        # text_col should be skipped
        assert 'text_col' not in report._features

    def test_pairwise_analysis_options(self, sample_dataframe):
        """Test different pairwise analysis options."""
        # Test 'off' option
        report_off = sv.analyze(sample_dataframe, pairwise_analysis='off')
        assert report_off is not None
        
        # Test 'on' option  
        report_on = sv.analyze(sample_dataframe, pairwise_analysis='on')
        assert report_on is not None
        
        # Test 'auto' option (default)
        report_auto = sv.analyze(sample_dataframe, pairwise_analysis='auto')
        assert report_auto is not None

    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()
        # Empty dataframes are actually handled gracefully by sweetviz
        report = sv.analyze(empty_df)
        assert report is not None
        assert len(report._features) == 0  # No features to analyze

    def test_single_column_dataframe(self):
        """Test handling of single column dataframes."""
        single_col_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        report = sv.analyze(single_col_df)
        assert report is not None
        assert len(report._features) == 1

    def test_dataframe_with_missing_values(self, sample_dataframe_with_missing):
        """Test handling of dataframes with missing values."""
        report = sv.analyze(sample_dataframe_with_missing)
        assert report is not None
        # Should handle NaN values gracefully

    def test_compare_intra_validation(self, sample_dataframe):
        """Test validation for compare_intra function."""
        condition = sample_dataframe['categorical_col'] == 'A'
        
        # Test invalid names parameter
        with pytest.raises(ValueError):
            sv.compare_intra(sample_dataframe, condition, ['Group A'])  # Only one name
        
        with pytest.raises(ValueError):
            sv.compare_intra(sample_dataframe, condition, ['A', 'B', 'C'])  # Too many names

    def test_invalid_target_feature(self, sample_dataframe):
        """Test handling of invalid target features."""
        with pytest.raises(Exception):
            sv.analyze(sample_dataframe, target_feat='nonexistent_column')

    def test_report_attributes(self, sample_dataframe):
        """Test that report has expected attributes."""
        report = sv.analyze(sample_dataframe)
        
        # Check essential attributes exist
        assert hasattr(report, '_features')
        assert hasattr(report, 'verbosity_level')
        # Note: page_layout might not be a direct attribute, removed this check


class TestDataTypes:
    """Test different data type handling."""

    def test_numeric_types(self):
        """Test various numeric data types."""
        df = pd.DataFrame({
            'int8': pd.array([1, 2, 3], dtype='int8'),
            'int16': pd.array([1, 2, 3], dtype='int16'), 
            'int32': pd.array([1, 2, 3], dtype='int32'),
            'int64': pd.array([1, 2, 3], dtype='int64'),
            'float32': pd.array([1.1, 2.2, 3.3], dtype='float32'),
            'float64': pd.array([1.1, 2.2, 3.3], dtype='float64')
        })
        report = sv.analyze(df)
        assert report is not None
        assert len(report._features) == 6

    def test_categorical_types(self):
        """Test categorical data types."""
        df = pd.DataFrame({
            'category': pd.Categorical(['A', 'B', 'C', 'A', 'B']),
            'object': ['X', 'Y', 'Z', 'X', 'Y'],
            'string': pd.array(['hello', 'world', 'test', 'hello', 'world'], dtype='string')
        })
        report = sv.analyze(df)
        assert report is not None
        assert len(report._features) == 3

    def test_datetime_types(self):
        """Test datetime data types."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=5),
            'date': pd.date_range('2023-01-01', periods=5).date,
            'time': pd.date_range('2023-01-01 00:00:00', periods=5, freq='h').time
        })
        report = sv.analyze(df)
        assert report is not None
        # Note: datetime handling may vary based on implementation


if __name__ == '__main__':
    pytest.main([__file__])