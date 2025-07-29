"""
Integration tests for AI features in the main sweetviz workflow.
"""
import pytest
import pandas as pd
import numpy as np
import sweetviz as sv


class TestAIIntegration:
    """Test AI features integrated into the main sweetviz workflow."""
    
    @pytest.fixture
    def sample_ai_data(self):
        """Create sample data designed to test AI features."""
        np.random.seed(42)
        n = 200
        
        return pd.DataFrame({
            'user_id': [f'USER{i:05d}' for i in range(n)],
            'email': [f'user{i}@{"company" if i%3==0 else "example"}.com' for i in range(n)],
            'age': np.random.normal(35, 12, n).astype(int),
            'income': np.random.lognormal(10, 0.8, n),
            'phone': [f'555-{i:04d}' for i in range(n)],
            'is_premium': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n),
            'signup_date': pd.date_range('2020-01-01', periods=n, freq='D')[:n],
            'active_flag': np.random.choice(['Y', 'N'], n, p=[0.8, 0.2]),
        })

    def test_ai_insights_in_analyze(self, sample_ai_data):
        """Test that AI insights are generated during analyze()."""
        report = sv.analyze(sample_ai_data, target_feat='is_premium')
        
        # Should have AI insights
        ai_insights = report.get_ai_insights()
        assert ai_insights is not None
        assert isinstance(ai_insights, dict)
        
        # Should have all major insight categories
        expected_categories = [
            'data_quality', 'distribution_insights', 'correlation_insights',
            'anomaly_detection', 'statistical_tests', 'target_insights', 'summary'
        ]
        for category in expected_categories:
            assert category in ai_insights
        
        # Should have semantic types
        semantic_types = report.get_semantic_types()
        assert semantic_types is not None
        assert isinstance(semantic_types, dict)
        assert len(semantic_types) == len(sample_ai_data.columns)

    def test_ai_insights_in_compare(self, sample_ai_data):
        """Test that AI insights work with compare()."""
        # Create a modified version for comparison
        compare_data = sample_ai_data.copy()
        compare_data['income'] = compare_data['income'] * 1.2  # Increase income
        compare_data['age'] = compare_data['age'] + 5  # Increase age
        
        report = sv.compare(
            [sample_ai_data, "Original"],
            [compare_data, "Modified"],
            target_feat='is_premium'
        )
        
        # Should still have AI insights
        ai_insights = report.get_ai_insights()
        assert ai_insights is not None
        
        # Should detect the comparison
        assert 'target_insights' in ai_insights

    def test_semantic_type_detection_accuracy(self, sample_ai_data):
        """Test accuracy of semantic type detection."""
        report = sv.analyze(sample_ai_data)
        semantic_types = report.get_semantic_types()
        
        # Test specific semantic type detections
        assert semantic_types['email']['semantic_type'] == 'email'
        assert semantic_types['email']['confidence'] > 0.8
        
        assert semantic_types['user_id']['semantic_type'] == 'identifier'
        assert semantic_types['user_id']['confidence'] > 0.5
        
        assert semantic_types['active_flag']['semantic_type'] == 'binary_flag'
        assert semantic_types['active_flag']['confidence'] > 0.8

    def test_ai_insights_quality_flags(self, sample_ai_data):
        """Test that AI insights detect data quality issues."""
        # Add some data quality issues
        problematic_data = sample_ai_data.copy()
        
        # Add missing values (>10% to trigger flag)
        missing_indices = np.random.choice(len(problematic_data), 25, replace=False)
        problematic_data.loc[missing_indices, 'income'] = np.nan
        
        # Add duplicates (>5% to trigger flag)
        duplicates = problematic_data.iloc[:15].copy()
        problematic_data = pd.concat([problematic_data, duplicates], ignore_index=True)
        
        report = sv.analyze(problematic_data)
        ai_insights = report.get_ai_insights()
        
        quality_flags = ai_insights['data_quality']['quality_flags']
        
        # Should detect high missing data
        assert 'HIGH_MISSING_DATA' in quality_flags
        
        # Should detect high duplicate rate
        assert 'HIGH_DUPLICATE_RATE' in quality_flags

    def test_target_relationship_analysis(self, sample_ai_data):
        """Test target relationship analysis."""
        report = sv.analyze(sample_ai_data, target_feat='is_premium')
        ai_insights = report.get_ai_insights()
        
        target_insights = ai_insights['target_insights']
        
        assert target_insights['target_column'] == 'is_premium'
        assert 'feature_relationships' in target_insights
        assert 'target_distribution' in target_insights
        
        # Should have analyzed relationships with other features
        relationships = target_insights['feature_relationships']
        assert len(relationships) > 0
        
        # Each relationship should have required fields
        for rel in relationships:
            assert 'feature' in rel
            assert 'relationship_type' in rel
            assert 'strength' in rel
            assert 'p_value' in rel
            assert 'significant' in rel

    def test_anomaly_detection_results(self, sample_ai_data):
        """Test anomaly detection functionality."""
        # Add some clear outliers
        outlier_data = sample_ai_data.copy()
        outlier_data.loc[0, 'age'] = 200  # Extreme outlier
        outlier_data.loc[1, 'income'] = 1000000  # Extreme outlier
        
        report = sv.analyze(outlier_data)
        ai_insights = report.get_ai_insights()
        
        anomaly_results = ai_insights['anomaly_detection']
        
        # Should have isolation forest results
        if 'isolation_forest' in anomaly_results:
            iso_results = anomaly_results['isolation_forest']
            assert 'anomaly_count' in iso_results
            assert 'anomaly_percentage' in iso_results
        
        # Should have statistical outlier detection
        if 'statistical_outliers' in anomaly_results:
            stat_outliers = anomaly_results['statistical_outliers']
            
            # Should detect outliers in age and income
            assert 'age' in stat_outliers
            assert 'income' in stat_outliers
            
            # Age outliers should be detected
            age_outliers = stat_outliers['age']
            assert age_outliers['outlier_count'] > 0

    def test_distribution_analysis(self, sample_ai_data):
        """Test distribution analysis functionality."""
        report = sv.analyze(sample_ai_data)
        ai_insights = report.get_ai_insights()
        
        dist_insights = ai_insights['distribution_insights']
        
        # Should analyze numeric columns
        numeric_cols = ['age', 'income']
        for col in numeric_cols:
            assert col in dist_insights
            
            col_analysis = dist_insights[col]
            assert 'mean' in col_analysis
            assert 'median' in col_analysis
            assert 'skewness' in col_analysis
            assert 'kurtosis' in col_analysis
            assert 'distribution_type' in col_analysis

    def test_ai_insights_with_missing_data(self):
        """Test AI insights handle missing data gracefully."""
        # Create data with significant missing values
        df_with_missing = pd.DataFrame({
            'col1': [1, 2, np.nan, np.nan, 5],
            'col2': ['a', np.nan, 'c', np.nan, 'e'],
            'col3': [np.nan, np.nan, np.nan, 4, 5],
        })
        
        report = sv.analyze(df_with_missing)
        ai_insights = report.get_ai_insights()
        
        # Should handle missing data without crashing
        assert ai_insights is not None
        assert 'data_quality' in ai_insights
        
        # Should detect high missing data percentage
        missing_pct = ai_insights['data_quality']['missing_data_percentage']
        assert missing_pct > 20  # Should be high due to many NaNs

    def test_ai_insights_verbosity_control(self, sample_ai_data):
        """Test that verbosity parameter controls AI insights output."""
        # This test mainly ensures no errors occur with different verbosity levels
        
        # Test with verbosity off
        report_quiet = sv.analyze(sample_ai_data, verbosity='off')
        assert report_quiet.get_ai_insights() is not None
        
        # Test with progress only
        report_progress = sv.analyze(sample_ai_data, verbosity='progress_only')
        assert report_progress.get_ai_insights() is not None
        
        # Test with full verbosity
        report_full = sv.analyze(sample_ai_data, verbosity='full')
        assert report_full.get_ai_insights() is not None

    def test_ai_insights_performance(self):
        """Test that AI insights don't significantly slow down analysis."""
        import time
        
        # Create a medium-sized dataset
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            f'col_{i}': np.random.randn(n) for i in range(10)
        })
        df['target'] = np.random.choice([0, 1], n)
        
        # Time the analysis
        start_time = time.time()
        report = sv.analyze(df, target_feat='target', verbosity='off')
        end_time = time.time()
        
        # Should complete within reasonable time (less than 30 seconds for 1000x10 data)
        elapsed_time = end_time - start_time
        assert elapsed_time < 30, f"Analysis took too long: {elapsed_time:.2f} seconds"
        
        # Should still generate AI insights
        assert report.get_ai_insights() is not None
        assert report.get_semantic_types() is not None


if __name__ == '__main__':
    pytest.main([__file__])