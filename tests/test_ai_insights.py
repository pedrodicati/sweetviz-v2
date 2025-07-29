"""
Tests for AI insights module.
"""
import pytest
import pandas as pd
import numpy as np
from sweetviz.ai_insights import DataInsightGenerator, SmartDataDetection


class TestDataInsightGenerator:
    """Test the DataInsightGenerator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'normal_dist': np.random.normal(100, 15, 1000),
            'skewed_dist': np.random.exponential(2, 1000),
            'categorical': np.random.choice(['A', 'B', 'C'], 1000),
            'binary_flag': np.random.choice([0, 1], 1000),
            'with_missing': [1, 2, np.nan, 4, 5] * 200,
            'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        })
    
    @pytest.fixture
    def insight_generator(self):
        """Create an insight generator instance."""
        return DataInsightGenerator(confidence_threshold=0.95)
    
    def test_generate_insights_basic(self, insight_generator, sample_data):
        """Test basic insight generation."""
        insights = insight_generator.generate_insights(sample_data)
        
        assert 'data_quality' in insights
        assert 'distribution_insights' in insights
        assert 'correlation_insights' in insights
        assert 'anomaly_detection' in insights
        assert 'statistical_tests' in insights
        assert 'summary' in insights
    
    def test_generate_insights_with_target(self, insight_generator, sample_data):
        """Test insight generation with target column."""
        insights = insight_generator.generate_insights(sample_data, target_col='target')
        
        assert 'target_insights' in insights
        assert insights['target_insights']['target_column'] == 'target'
        assert 'feature_relationships' in insights['target_insights']
    
    def test_data_quality_assessment(self, insight_generator, sample_data):
        """Test data quality assessment."""
        quality = insight_generator._assess_data_quality(sample_data)
        
        assert 'total_rows' in quality
        assert 'total_columns' in quality
        assert 'missing_data_percentage' in quality
        assert 'duplicate_rows' in quality
        assert 'column_quality' in quality
        assert 'quality_flags' in quality
        
        assert quality['total_rows'] == 1000
        assert quality['total_columns'] == 6
        assert quality['missing_data_percentage'] > 0  # Due to with_missing column
    
    def test_distribution_analysis(self, insight_generator, sample_data):
        """Test distribution analysis."""
        distributions = insight_generator._analyze_distributions(sample_data)
        
        # Should analyze numeric columns
        assert 'normal_dist' in distributions
        assert 'skewed_dist' in distributions
        assert 'binary_flag' in distributions
        
        # Should have distribution metrics
        normal_analysis = distributions['normal_dist']
        assert 'mean' in normal_analysis
        assert 'median' in normal_analysis
        assert 'skewness' in normal_analysis
        assert 'kurtosis' in normal_analysis
        assert 'distribution_type' in normal_analysis
    
    def test_correlation_analysis(self, insight_generator, sample_data):
        """Test correlation analysis."""
        correlations = insight_generator._analyze_correlations(sample_data)
        
        assert 'correlation_matrix' in correlations
        assert 'high_correlations' in correlations
    
    def test_anomaly_detection(self, insight_generator, sample_data):
        """Test anomaly detection."""
        anomalies = insight_generator._detect_anomalies(sample_data)
        
        assert 'isolation_forest' in anomalies or 'message' in anomalies
        assert 'statistical_outliers' in anomalies or 'message' in anomalies
    
    def test_statistical_tests(self, insight_generator, sample_data):
        """Test statistical hypothesis tests."""
        tests = insight_generator._perform_statistical_tests(sample_data)
        
        # Should contain some test results
        assert isinstance(tests, dict)
    
    def test_target_analysis(self, insight_generator, sample_data):
        """Test target relationship analysis."""
        target_insights = insight_generator._analyze_target_relationships(sample_data, 'target')
        
        assert 'target_column' in target_insights
        assert 'target_type' in target_insights
        assert 'target_distribution' in target_insights
        assert 'feature_relationships' in target_insights
        
        assert target_insights['target_column'] == 'target'
    
    def test_empty_dataframe(self, insight_generator):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        insights = insight_generator.generate_insights(empty_df)
        
        assert 'data_quality' in insights
        assert insights['data_quality']['total_rows'] == 0
        assert insights['data_quality']['total_columns'] == 0


class TestSmartDataDetection:
    """Test the SmartDataDetection class."""
    
    @pytest.fixture
    def smart_detector(self):
        """Create a smart data detection instance."""
        return SmartDataDetection()
    
    @pytest.fixture
    def semantic_data(self):
        """Create data with various semantic types."""
        return pd.DataFrame({
            'emails': ['user1@example.com', 'user2@test.org', 'admin@company.net'],
            'phone_numbers': ['1234567890', '555-123-4567', '+1-800-555-0123'],
            'urls': ['https://example.com', 'http://test.org', 'https://company.net/page'],
            'ids': ['USR001', 'USR002', 'USR003'],
            'binary_flags': ['Y', 'N', 'Y'],
            'dates': ['2023-01-01', '2023-02-15', '2023-12-31'],
            'normal_text': ['hello', 'world', 'test'],
        })
    
    def test_detect_semantic_types(self, smart_detector, semantic_data):
        """Test semantic type detection."""
        types = smart_detector.detect_semantic_types(semantic_data)
        
        assert len(types) == len(semantic_data.columns)
        
        # Check that all columns have required fields
        for col_name, analysis in types.items():
            assert 'pandas_type' in analysis
            assert 'semantic_type' in analysis
            assert 'confidence' in analysis
            assert 'characteristics' in analysis
    
    def test_email_detection(self, smart_detector):
        """Test email address detection."""
        email_series = pd.Series(['user@example.com', 'admin@test.org', 'invalid-email'])
        confidence = smart_detector._is_email_column(email_series, smart_detector.email_pattern)
        
        assert 0 < confidence < 1  # Should detect some but not all as emails
    
    def test_phone_detection(self, smart_detector):
        """Test phone number detection."""
        phone_series = pd.Series(['1234567890', '555-123-4567', 'not-a-phone'])
        confidence = smart_detector._is_phone_column(phone_series, smart_detector.phone_pattern)
        
        assert confidence >= 0  # Should work without errors
    
    def test_binary_flag_detection(self, smart_detector):
        """Test binary flag detection."""
        binary_series = pd.Series(['Y', 'N', 'Y', 'N'])
        confidence = smart_detector._is_binary_flag(binary_series)
        
        assert confidence > 0.5  # Should detect as binary
    
    def test_identifier_detection(self, smart_detector):
        """Test identifier detection."""
        id_series = pd.Series(['ID001', 'ID002', 'ID003', 'ID004'])
        confidence = smart_detector._is_identifier_column(id_series)
        
        assert confidence > 0  # Should detect some identifier characteristics
    
    def test_date_string_detection(self, smart_detector):
        """Test date string detection."""
        date_series = pd.Series(['2023-01-01', '2023-02-15', 'not-a-date'])
        confidence = smart_detector._is_date_string_column(date_series)
        
        assert 0 < confidence < 1  # Should detect some as dates
    
    def test_empty_series(self, smart_detector):
        """Test handling of empty series."""
        empty_series = pd.Series([], dtype=object)
        analysis = smart_detector._analyze_column(empty_series)
        
        assert analysis['confidence'] == 0.0
        assert analysis['semantic_type'] == 'unknown'


class TestIntegration:
    """Integration tests for AI insights functionality."""
    
    def test_insights_with_real_world_data(self):
        """Test insights generation with realistic data."""
        # Create a more realistic dataset
        np.random.seed(42)
        n = 500
        
        df = pd.DataFrame({
            'user_id': [f'USR{i:04d}' for i in range(n)],
            'age': np.random.normal(35, 12, n).astype(int),
            'income': np.random.lognormal(10, 0.8, n),
            'email': [f'user{i}@example.com' for i in range(n)],
            'is_premium': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'signup_date': pd.date_range('2020-01-01', periods=n, freq='D')[:n],
            'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        })
        
        # Add some missing values and outliers
        df.loc[np.random.choice(n, 20, replace=False), 'income'] = np.nan
        df.loc[np.random.choice(n, 5, replace=False), 'age'] = 150  # Outliers
        
        generator = DataInsightGenerator()
        insights = generator.generate_insights(df, target_col='is_premium')
        
        # Should generate comprehensive insights
        assert len(insights) >= 6  # All major insight categories
        assert len(insights['summary']) > 0  # Should have summary
        assert 'target_insights' in insights  # Should analyze target
        
        # Should detect data quality issues
        quality_flags = insights['data_quality']['quality_flags']
        assert isinstance(quality_flags, list)
        
        # Should detect semantic types
        detector = SmartDataDetection()
        semantic_types = detector.detect_semantic_types(df)
        
        # Should identify email column
        email_analysis = semantic_types['email']
        assert email_analysis['semantic_type'] == 'email'
        assert email_analysis['confidence'] > 0.8
        
        # Should identify ID column
        id_analysis = semantic_types['user_id']
        assert id_analysis['semantic_type'] == 'identifier'
        assert id_analysis['confidence'] > 0.5


if __name__ == '__main__':
    pytest.main([__file__])