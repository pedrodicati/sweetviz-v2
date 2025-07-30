"""
Test suite for Phase 5: Advanced AI Features and MLOps Integrations
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json


class TestMLOpsIntegrations:
    """Test MLOps platform integrations"""

    def setup_method(self):
        """Set up test data"""
        self.test_df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5, np.nan],
            'categorical_col': ['A', 'B', 'A', 'C', 'B', 'A'],
            'text_col': ['hello', 'world', 'test', 'data', 'science', 'ml']
        })
        
        self.test_report_data = {
            'dataset_info': {
                'shape': (6, 3),
                'num_columns': 3,
                'missing_values': 1,
                'data_types': {'float64': 1, 'object': 2},
                'memory_usage_mb': 0.1
            },
            'summary_stats': {
                'total_features': 3,
                'numeric_features': 1,
                'categorical_features': 2,
                'missing_percentage': 5.56
            },
            'feature_analysis': {},
            'correlations': {},
            'metadata': {
                'generation_time': 1234567890,
                'sweetviz_version': '2.3.1',
                'report_type': 'analyze'
            }
        }

    def test_mlflow_integration_import(self):
        """Test that MLflow integration can be imported"""
        from sweetviz.mlops_integrations import MLflowIntegration
        integration = MLflowIntegration()
        assert integration is not None

    def test_wandb_integration_import(self):
        """Test that WandB integration can be imported"""
        from sweetviz.mlops_integrations import WandBIntegration
        integration = WandBIntegration()
        assert integration is not None

    def test_mlops_manager_import(self):
        """Test that MLOps manager can be imported"""
        from sweetviz.mlops_integrations import get_mlops_manager
        manager = get_mlops_manager()
        assert manager is not None

    def test_mlflow_availability_without_package(self):
        """Test MLflow availability when package not installed"""
        from sweetviz.mlops_integrations import MLflowIntegration
        
        with patch('sweetviz.mlops_integrations.MLflowIntegration._get_mlflow') as mock_get:
            mock_get.side_effect = ImportError("MLflow not installed")
            integration = MLflowIntegration()
            assert not integration.is_available()

    def test_wandb_availability_without_package(self):
        """Test WandB availability when package not installed"""
        from sweetviz.mlops_integrations import WandBIntegration
        
        with patch('sweetviz.mlops_integrations.WandBIntegration._get_wandb') as mock_get:
            mock_get.side_effect = ImportError("wandb not installed")
            integration = WandBIntegration()
            assert not integration.is_available()

    @patch('sweetviz.mlops_integrations.MLflowIntegration.is_available')
    def test_mlflow_export_not_available(self, mock_available):
        """Test MLflow export when not available"""
        mock_available.return_value = False
        
        from sweetviz.mlops_integrations import MLflowIntegration
        integration = MLflowIntegration()
        
        result = integration.export_report(self.test_report_data)
        assert 'error' in result
        assert 'not available' in result['error']

    @patch('sweetviz.mlops_integrations.WandBIntegration.is_available')
    def test_wandb_export_not_available(self, mock_available):
        """Test WandB export when not available"""
        mock_available.return_value = False
        
        from sweetviz.mlops_integrations import WandBIntegration
        integration = WandBIntegration()
        
        result = integration.export_report(self.test_report_data)
        assert 'error' in result
        assert 'not available' in result['error']

    def test_mlops_manager_list_integrations(self):
        """Test listing available integrations"""
        from sweetviz.mlops_integrations import MLOpsManager
        manager = MLOpsManager()
        
        # Should return empty list when no integrations available
        available = manager.list_available_integrations()
        assert isinstance(available, list)

    def test_dataframe_report_mlops_methods(self):
        """Test that DataframeReport has MLOps export methods"""
        import sweetviz as sv
        
        # Create a simple report
        report = sv.analyze(self.test_df)
        
        # Test methods exist
        assert hasattr(report, 'to_mlflow')
        assert hasattr(report, 'to_wandb')
        assert hasattr(report, '_extract_report_data')

    def test_dataframe_report_extract_data(self):
        """Test report data extraction"""
        import sweetviz as sv
        
        report = sv.analyze(self.test_df)
        extracted_data = report._extract_report_data()
        
        assert 'dataset_info' in extracted_data
        assert 'summary_stats' in extracted_data
        assert 'feature_analysis' in extracted_data
        assert 'metadata' in extracted_data
        
        # Check dataset info
        assert extracted_data['dataset_info']['shape'] == self.test_df.shape
        assert extracted_data['dataset_info']['num_columns'] == len(self.test_df.columns)

    def test_dataframe_report_mlflow_export_no_dependency(self):
        """Test MLflow export when dependency not available"""
        import sweetviz as sv
        
        report = sv.analyze(self.test_df)
        
        with patch('sweetviz.mlops_integrations.get_mlops_manager') as mock_get:
            mock_manager = Mock()
            mock_manager.export_to_mlflow.return_value = {"error": "MLflow not available"}
            mock_get.return_value = mock_manager
            
            result = report.to_mlflow()
            assert 'error' in result

    def test_dataframe_report_wandb_export_no_dependency(self):
        """Test WandB export when dependency not available"""
        import sweetviz as sv
        
        report = sv.analyze(self.test_df)
        
        with patch('sweetviz.mlops_integrations.get_mlops_manager') as mock_get:
            mock_manager = Mock()
            mock_manager.export_to_wandb.return_value = {"error": "WandB not available"}
            mock_get.return_value = mock_manager
            
            result = report.to_wandb()
            assert 'error' in result


class TestEnhancedAIInsights:
    """Test enhanced AI insights for Phase 5"""

    def setup_method(self):
        """Set up test data"""
        self.test_df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 120, np.nan],  # Include outlier
            'salary': [50000, 60000, 70000, 80000, 90000, 200000, 55000],
            'department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT'],
            'performance': [3.2, 4.1, 3.8, 4.5, 3.9, 4.2, 3.6]
        })

    def test_ai_manager_import(self):
        """Test that AI manager can be imported"""
        from sweetviz.ai_insights import get_ai_manager
        manager = get_ai_manager()
        assert manager is not None

    def test_enhanced_data_summary_no_ai(self):
        """Test enhanced data summary without AI configured"""
        from sweetviz.ai_insights import OpenAIInsightProvider
        
        provider = OpenAIInsightProvider()  # No API key
        summary = provider.generate_data_summary(self.test_df)
        
        assert isinstance(summary, str)
        assert 'not available' in summary or 'not configured' in summary

    def test_enhanced_anomaly_detection(self):
        """Test enhanced anomaly detection"""
        from sweetviz.ai_insights import OpenAIInsightProvider
        
        provider = OpenAIInsightProvider()
        anomalies = provider.detect_anomalies(self.test_df)
        
        assert isinstance(anomalies, dict)
        assert 'statistical_outliers' in anomalies
        assert 'pattern_anomalies' in anomalies
        assert 'recommendations' in anomalies
        
        # Should detect outlier in age column
        if 'age' in anomalies['statistical_outliers']:
            age_outliers = anomalies['statistical_outliers']['age']
            assert 'iqr_outliers' in age_outliers
            assert age_outliers['iqr_outliers']['count'] > 0

    def test_enhanced_correlation_explanation(self):
        """Test enhanced correlation explanation"""
        from sweetviz.ai_insights import OpenAIInsightProvider
        
        provider = OpenAIInsightProvider()
        correlations = {
            'age_salary': 0.85,
            'age_performance': 0.65,
            'salary_performance': 0.45
        }
        
        explanation = provider.explain_correlations(correlations)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert 'strong' in explanation.lower() or 'correlation' in explanation.lower()

    def test_anomaly_detection_pattern_analysis(self):
        """Test pattern-based anomaly detection"""
        # Create test data with various anomalies
        anomaly_df = pd.DataFrame({
            'constant_col': [1, 1, 1, 1, 1],  # Constant column
            'mostly_missing': [1, np.nan, np.nan, np.nan, np.nan],  # High missing
            'normal_col': [1, 2, 3, 4, 5],
            'numeric_as_string': ['1', '2', '3', '4', '5']  # Numbers as strings
        })
        
        # Add duplicate rows
        anomaly_df = pd.concat([anomaly_df, anomaly_df.iloc[[0, 1]]], ignore_index=True)
        
        from sweetviz.ai_insights import OpenAIInsightProvider
        provider = OpenAIInsightProvider()
        anomalies = provider.detect_anomalies(anomaly_df)
        
        # Should detect constant columns
        if 'pattern_anomalies' in anomalies:
            pattern_anomalies = anomalies['pattern_anomalies']
            
            # Check for constant column detection
            if 'constant_columns' in pattern_anomalies:
                assert 'constant_col' in pattern_anomalies['constant_columns']['columns']
            
            # Check for high missing data detection
            if 'high_missing_columns' in pattern_anomalies:
                high_missing = pattern_anomalies['high_missing_columns']['columns']
                assert any(col['column'] == 'mostly_missing' for col in high_missing)
            
            # Check for duplicate rows
            if 'duplicate_rows' in pattern_anomalies:
                assert pattern_anomalies['duplicate_rows']['count'] > 0

    @patch('sweetviz.ai_insights.OpenAIInsightProvider.is_available')
    @patch('sweetviz.ai_insights.OpenAIInsightProvider._get_client')
    def test_ai_summary_with_mocked_client(self, mock_get_client, mock_is_available):
        """Test AI summary with mocked OpenAI client"""
        # Mock availability
        mock_is_available.return_value = True
        
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "This dataset contains employee information with 7 rows and 4 columns. There are numeric features like age and salary, and categorical features like department. Some data quality issues detected include potential outliers in age."
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        from sweetviz.ai_insights import OpenAIInsightProvider
        provider = OpenAIInsightProvider(api_key="test-key")
        
        summary = provider.generate_data_summary(self.test_df)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'dataset' in summary.lower()


class TestNaturalLanguageQuery:
    """Test natural language query interface"""

    def setup_method(self):
        """Set up test data"""
        self.test_df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
            'performance_score': [3.2, 4.1, 3.8, 4.5, 3.9]
        })

    def test_nl_query_interface_import(self):
        """Test that natural language query interface can be imported"""
        from sweetviz.nl_query import get_nl_query_interface, ask_question, get_query_suggestions
        
        interface = get_nl_query_interface()
        assert interface is not None
        assert callable(ask_question)
        assert callable(get_query_suggestions)

    def test_rule_based_parser_import(self):
        """Test that rule-based parser can be imported"""
        from sweetviz.nl_query import RuleBasedQueryParser
        
        parser = RuleBasedQueryParser()
        assert parser.is_available()

    def test_query_suggestions(self):
        """Test getting query suggestions"""
        from sweetviz.nl_query import get_query_suggestions
        
        suggestions = get_query_suggestions()
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any('mean' in suggestion for suggestion in suggestions)

    def test_simple_mean_query(self):
        """Test simple mean query"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("mean of age", self.test_df)
        
        assert 'result' in result
        assert 'mean' in result['result']
        assert result['result']['mean'] == self.test_df['age'].mean()

    def test_simple_max_query(self):
        """Test simple max query"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("maximum of salary", self.test_df)
        
        assert 'result' in result
        assert 'max' in result['result']
        assert result['result']['max'] == self.test_df['salary'].max()

    def test_distribution_query(self):
        """Test distribution query"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("distribution of age", self.test_df)
        
        assert 'result' in result
        assert 'column' in result['result']
        assert result['result']['column'] == 'age'
        assert 'type' in result['result']

    def test_missing_values_query(self):
        """Test missing values query"""
        # Add some missing values
        test_df_with_missing = self.test_df.copy()
        test_df_with_missing.loc[0, 'age'] = np.nan
        
        from sweetviz.nl_query import ask_question
        
        result = ask_question("missing values in age", test_df_with_missing)
        
        assert 'result' in result
        assert 'missing_count' in result['result']
        assert result['result']['missing_count'] == 1

    def test_correlation_query(self):
        """Test correlation query"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("correlation between age and salary", self.test_df)
        
        assert 'result' in result
        assert 'correlation' in result['result']
        assert isinstance(result['result']['correlation'], float)

    def test_data_shape_query(self):
        """Test data shape query"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("shape of dataset", self.test_df)
        
        assert 'result' in result
        assert 'shape' in result['result']
        assert result['result']['shape'] == self.test_df.shape

    def test_unique_values_query(self):
        """Test unique values query"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("unique values in department", self.test_df)
        
        assert 'result' in result
        assert 'unique_count' in result['result']
        assert result['result']['unique_count'] == self.test_df['department'].nunique()

    def test_column_info_query(self):
        """Test column info query"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("info about age", self.test_df)
        
        assert 'result' in result
        assert 'column' in result['result']
        assert 'dtype' in result['result']

    def test_top_n_query(self):
        """Test top N query"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("top 3 salary", self.test_df)
        
        assert 'result' in result
        assert 'values' in result['result']
        assert len(result['result']['values']) == 3

    def test_unknown_query(self):
        """Test unknown/unsupported query"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("complex unsupported query", self.test_df)
        
        assert 'error' in result or 'suggestions' in result

    def test_column_not_found_query(self):
        """Test query with non-existent column"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("mean of nonexistent_column", self.test_df)
        
        assert 'result' in result
        assert 'error' in result['result']
        assert 'not found' in result['result']['error']

    def test_non_numeric_operation_on_text(self):
        """Test numeric operation on non-numeric column"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("mean of department", self.test_df)
        
        assert 'result' in result
        assert 'error' in result['result']
        assert 'not numeric' in result['result']['error']

    def test_ai_parser_availability(self):
        """Test AI parser availability"""
        from sweetviz.nl_query import AIQueryParser
        
        parser = AIQueryParser()  # No API key
        assert not parser.is_available()
        
        parser_with_key = AIQueryParser(api_key="test-key")
        # Should still be False because openai package check
        # In a real test with openai installed, this would check differently

    def test_case_insensitive_column_matching(self):
        """Test case-insensitive column matching"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("mean of AGE", self.test_df)
        
        assert 'result' in result
        assert 'mean' in result['result']
        assert result['result']['column'] == 'age'  # Should find 'age' column

    def test_partial_column_matching(self):
        """Test partial column name matching"""
        from sweetviz.nl_query import ask_question
        
        result = ask_question("mean of performance", self.test_df)
        
        assert 'result' in result
        assert 'mean' in result['result']
        assert 'performance_score' in result['result']['column']


class TestPhase5Integration:
    """Test integration of all Phase 5 features"""

    def setup_method(self):
        """Set up test data"""
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 100],  # With outlier
            'feature2': [10, 20, 30, 40, 50, 60],
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1, 0, 1]
        })

    def test_all_phase5_imports(self):
        """Test that all Phase 5 features can be imported from main module"""
        import sweetviz as sv
        
        # Test MLOps manager
        assert hasattr(sv, 'get_mlops_manager')
        manager = sv.get_mlops_manager()
        assert manager is not None
        
        # Test natural language queries
        assert hasattr(sv, 'ask_question')
        assert hasattr(sv, 'get_query_suggestions')
        assert hasattr(sv, 'get_nl_query_interface')
        
        # Test functions are callable
        assert callable(sv.ask_question)
        assert callable(sv.get_query_suggestions)
        assert callable(sv.get_nl_query_interface)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow with Phase 5 features"""
        import sweetviz as sv
        
        # 1. Create analysis
        report = sv.analyze(self.test_df)
        assert report is not None
        
        # 2. Test MLOps export methods exist
        assert hasattr(report, 'to_mlflow')
        assert hasattr(report, 'to_wandb')
        
        # 3. Test natural language queries
        result = sv.ask_question("mean of feature1", self.test_df)
        assert 'result' in result
        
        # 4. Test AI manager
        ai_manager = sv.get_ai_manager()
        assert ai_manager is not None
        
        # 5. Test MLOps manager
        mlops_manager = sv.get_mlops_manager()
        assert mlops_manager is not None

    def test_enhanced_ai_with_report_integration(self):
        """Test enhanced AI insights with report integration"""
        import sweetviz as sv
        
        # Create report
        report = sv.analyze(self.test_df)
        
        # Extract report data (used by MLOps exports)
        report_data = report._extract_report_data()
        
        assert 'dataset_info' in report_data
        assert 'summary_stats' in report_data
        assert report_data['dataset_info']['shape'] == self.test_df.shape
        
        # Test AI manager with the dataset
        ai_manager = sv.get_ai_manager()
        
        # Even without API keys, should not crash
        summary = ai_manager.generate_data_summary(self.test_df)
        assert isinstance(summary, (str, type(None)))
        
        anomalies = ai_manager.detect_anomalies(self.test_df)
        assert isinstance(anomalies, (dict, type(None)))

    def test_backwards_compatibility_with_phase5(self):
        """Test that Phase 5 doesn't break existing functionality"""
        import sweetviz as sv
        
        # Traditional sweetviz workflow should still work
        report = sv.analyze(self.test_df)
        assert report is not None
        
        # Generate HTML (this should work without Phase 5 dependencies)
        html_output = report.to_html()
        assert isinstance(html_output, str)
        assert len(html_output) > 0
        
        # Compare functionality
        train_df = self.test_df.iloc[:3]
        test_df = self.test_df.iloc[3:]
        
        comparison_report = sv.compare([train_df, "Train"], [test_df, "Test"])
        assert comparison_report is not None