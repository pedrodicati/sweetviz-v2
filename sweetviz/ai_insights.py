"""
AI-powered data insights module for Sweetviz v2.

This module provides automated data insight generation, anomaly detection,
and intelligent data quality assessment using modern AI/ML techniques.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings


class DataInsightGenerator:
    """
    Generate automated insights about datasets using AI/ML techniques.
    
    This class provides methods to automatically detect patterns, anomalies,
    and generate natural language insights about data characteristics.
    """

    def __init__(self, confidence_threshold: float = 0.95):
        """
        Initialize the insight generator.
        
        Args:
            confidence_threshold: Statistical confidence level for insights
        """
        self.confidence_threshold = confidence_threshold
        self.insights = []

    def generate_insights(
        self, 
        df: pd.DataFrame, 
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive insights for a DataFrame.
        
        Args:
            df: Input DataFrame to analyze
            target_col: Optional target column for supervised insights
            
        Returns:
            Dictionary containing various types of insights
        """
        insights = {
            'data_quality': self._assess_data_quality(df),
            'distribution_insights': self._analyze_distributions(df),
            'correlation_insights': self._analyze_correlations(df),
            'anomaly_detection': self._detect_anomalies(df),
            'statistical_tests': self._perform_statistical_tests(df),
        }
        
        if target_col and target_col in df.columns:
            insights['target_insights'] = self._analyze_target_relationships(df, target_col)
            
        # Generate natural language summaries
        insights['summary'] = self._generate_summary(insights, df.shape)
        
        return insights

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality metrics."""
        quality_metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': (df.isnull().sum().sum() / df.size) * 100 if df.size > 0 else 0.0,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0.0,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }
        
        # Column-specific quality metrics
        column_quality = {}
        for col in df.columns:
            column_quality[col] = {
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_values': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100,
                'data_type': str(df[col].dtype),
            }
            
        quality_metrics['column_quality'] = column_quality
        
        # Quality flags
        quality_flags = []
        if quality_metrics['missing_data_percentage'] > 10:
            quality_flags.append("HIGH_MISSING_DATA")
        if quality_metrics['duplicate_percentage'] > 5:
            quality_flags.append("HIGH_DUPLICATE_RATE")
        if quality_metrics['memory_usage_mb'] > 100:
            quality_flags.append("LARGE_MEMORY_USAGE")
            
        quality_metrics['quality_flags'] = quality_flags
        
        return quality_metrics

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical distributions of numeric columns."""
        distribution_insights = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:  # Skip columns with too few values
                continue
                
            insights = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'skewness': stats.skew(col_data),
                'kurtosis': stats.kurtosis(col_data),
                'range': col_data.max() - col_data.min(),
            }
            
            # Test for normality
            if len(col_data) >= 20:  # Minimum sample size for normality test
                shapiro_stat, shapiro_p = stats.shapiro(col_data[:5000])  # Limit sample size
                insights['normality_test'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > (1 - self.confidence_threshold),
                }
            
            # Distribution characteristics
            insights['distribution_type'] = self._classify_distribution(insights)
            
            distribution_insights[col] = insights
            
        return distribution_insights

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations and relationships between variables."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'message': 'Insufficient numeric columns for correlation analysis'}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': 'very_strong' if abs(corr_val) > 0.9 else 'strong'
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations,
            'max_correlation': corr_matrix.abs().unstack().sort_values(ascending=False).iloc[1] if len(corr_matrix) > 1 else None,
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using isolation forest and statistical methods."""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if numeric_df.shape[1] == 0 or numeric_df.shape[0] < 10:
            return {'message': 'Insufficient data for anomaly detection'}
        
        anomaly_results = {}
        
        # Isolation Forest for multivariate anomaly detection
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_data)
            
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            anomaly_percentage = (len(anomaly_indices) / len(numeric_df)) * 100
            
            anomaly_results['isolation_forest'] = {
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': anomaly_percentage,
                'anomaly_indices': anomaly_indices.tolist()[:50],  # Limit output size
            }
        except Exception as e:
            anomaly_results['isolation_forest'] = {'error': str(e)}
        
        # Statistical outlier detection for individual columns
        column_outliers = {}
        for col in numeric_df.columns:
            col_data = numeric_df[col]
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            column_outliers[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(col_data)) * 100,
                'bounds': {'lower': lower_bound, 'upper': upper_bound},
            }
        
        anomaly_results['statistical_outliers'] = column_outliers
        
        return anomaly_results

    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform various statistical hypothesis tests."""
        test_results = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Test for independence between categorical variables
        if len(categorical_cols) >= 2:
            cat_independence_tests = []
            for i, col1 in enumerate(categorical_cols[:5]):  # Limit to first 5 to avoid excessive computation
                for col2 in categorical_cols[i+1:6]:
                    try:
                        contingency_table = pd.crosstab(df[col1].fillna('missing'), df[col2].fillna('missing'))
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                        
                        cat_independence_tests.append({
                            'variable1': col1,
                            'variable2': col2,
                            'chi2_statistic': chi2,
                            'p_value': p_value,
                            'degrees_of_freedom': dof,
                            'is_independent': p_value > (1 - self.confidence_threshold),
                        })
                    except Exception:
                        continue  # Skip if test fails
            
            test_results['categorical_independence'] = cat_independence_tests
        
        # Variance equality tests for numeric variables
        if len(numeric_cols) >= 2:
            variance_tests = []
            for i, col1 in enumerate(numeric_cols[:5]):
                for col2 in numeric_cols[i+1:6]:
                    try:
                        data1 = df[col1].dropna()
                        data2 = df[col2].dropna()
                        
                        if len(data1) >= 10 and len(data2) >= 10:
                            f_stat, p_value = stats.levene(data1, data2)
                            
                            variance_tests.append({
                                'variable1': col1,
                                'variable2': col2,
                                'f_statistic': f_stat,
                                'p_value': p_value,
                                'equal_variances': p_value > (1 - self.confidence_threshold),
                            })
                    except Exception:
                        continue
            
            test_results['variance_equality'] = variance_tests
        
        return test_results

    def _analyze_target_relationships(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze relationships between features and target variable."""
        target_insights = {
            'target_column': target_col,
            'target_type': str(df[target_col].dtype),
            'target_distribution': {},
        }
        
        # Target distribution analysis
        if df[target_col].dtype in ['int64', 'float64']:
            target_insights['target_distribution'] = {
                'mean': df[target_col].mean(),
                'median': df[target_col].median(),
                'std': df[target_col].std(),
                'min': df[target_col].min(),
                'max': df[target_col].max(),
            }
        else:
            target_insights['target_distribution'] = df[target_col].value_counts().to_dict()
        
        # Feature importance based on correlation/association
        feature_relationships = []
        
        for col in df.columns:
            if col == target_col:
                continue
                
            try:
                if df[col].dtype in ['int64', 'float64'] and df[target_col].dtype in ['int64', 'float64']:
                    # Pearson correlation for numeric-numeric
                    corr, p_value = stats.pearsonr(df[col].dropna(), df[target_col].dropna())
                    feature_relationships.append({
                        'feature': col,
                        'relationship_type': 'correlation',
                        'strength': abs(corr),
                        'p_value': p_value,
                        'significant': p_value < (1 - self.confidence_threshold),
                    })
                elif df[col].dtype not in ['int64', 'float64'] and df[target_col].dtype not in ['int64', 'float64']:
                    # Chi-square test for categorical-categorical
                    contingency_table = pd.crosstab(df[col].fillna('missing'), df[target_col].fillna('missing'))
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    feature_relationships.append({
                        'feature': col,
                        'relationship_type': 'chi_square',
                        'strength': chi2 / (chi2 + contingency_table.sum().sum()),  # Normalized chi-square
                        'p_value': p_value,
                        'significant': p_value < (1 - self.confidence_threshold),
                    })
            except Exception:
                continue  # Skip if calculation fails
        
        # Sort by relationship strength
        feature_relationships.sort(key=lambda x: x['strength'], reverse=True)
        target_insights['feature_relationships'] = feature_relationships[:10]  # Top 10
        
        return target_insights

    def _classify_distribution(self, insights: Dict[str, Any]) -> str:
        """Classify the type of distribution based on statistical properties."""
        skewness = insights.get('skewness', 0)
        kurtosis = insights.get('kurtosis', 0)
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return 'approximately_normal'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        elif kurtosis > 3:
            return 'heavy_tailed'
        elif kurtosis < -1:
            return 'light_tailed'
        else:
            return 'irregular'

    def _generate_summary(self, insights: Dict[str, Any], shape: Tuple[int, int]) -> List[str]:
        """Generate natural language summary of key insights."""
        summary = []
        
        # Data quality summary
        quality = insights.get('data_quality', {})
        missing_pct = quality.get('missing_data_percentage', 0)
        
        summary.append(f"Dataset contains {shape[0]:,} rows and {shape[1]} columns.")
        
        if missing_pct > 10:
            summary.append(f"⚠️ High missing data rate: {missing_pct:.1f}% of values are missing.")
        elif missing_pct > 0:
            summary.append(f"Missing data: {missing_pct:.1f}% of values are missing.")
        else:
            summary.append("✅ No missing data detected.")
        
        # Correlation insights
        corr_insights = insights.get('correlation_insights', {})
        high_corrs = corr_insights.get('high_correlations', [])
        
        if high_corrs:
            summary.append(f"🔗 Found {len(high_corrs)} strong correlations between features.")
            strongest = max(high_corrs, key=lambda x: abs(x['correlation']))
            summary.append(f"Strongest correlation: {strongest['feature1']} ↔ {strongest['feature2']} (r={strongest['correlation']:.3f})")
        
        # Anomaly insights
        anomaly_insights = insights.get('anomaly_detection', {})
        iso_forest = anomaly_insights.get('isolation_forest', {})
        
        if 'anomaly_percentage' in iso_forest:
            anomaly_pct = iso_forest['anomaly_percentage']
            if anomaly_pct > 5:
                summary.append(f"⚠️ Potential data quality issue: {anomaly_pct:.1f}% of rows flagged as anomalies.")
            elif anomaly_pct > 0:
                summary.append(f"🔍 {anomaly_pct:.1f}% of rows identified as potential anomalies.")
        
        # Distribution insights
        dist_insights = insights.get('distribution_insights', {})
        normal_cols = [col for col, data in dist_insights.items() 
                      if data.get('distribution_type') == 'approximately_normal']
        
        if normal_cols:
            summary.append(f"📊 {len(normal_cols)} numeric columns have approximately normal distributions.")
        
        return summary


class SmartDataDetection:
    """
    Enhanced data type detection using ML techniques.
    
    Provides intelligent detection of data types beyond pandas' basic inference,
    including semantic types like emails, phone numbers, IDs, etc.
    """
    
    def __init__(self):
        """Initialize the smart detection engine."""
        self.email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        self.phone_pattern = r'^[\+]?[1-9]?[0-9]{7,15}$'
        self.url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        
    def detect_semantic_types(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Detect semantic data types for each column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to detected semantic types and confidence
        """
        semantic_types = {}
        
        for col in df.columns:
            col_analysis = self._analyze_column(df[col])
            semantic_types[col] = col_analysis
            
        return semantic_types
    
    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single column for semantic type detection."""
        import re
        
        analysis = {
            'pandas_type': str(series.dtype),
            'semantic_type': 'unknown',
            'confidence': 0.0,
            'characteristics': {},
        }
        
        # Skip if mostly null
        if len(series) == 0 or series.isnull().sum() / len(series) > 0.8:
            analysis['semantic_type'] = 'mostly_null'
            analysis['confidence'] = 0.9
            return analysis
        
        # Convert to string for pattern matching
        str_series = series.astype(str).dropna()
        
        if len(str_series) == 0:
            return analysis
        
        # Check for various semantic types
        type_checks = [
            ('email', self._is_email_column, self.email_pattern),
            ('phone_number', self._is_phone_column, self.phone_pattern),
            ('url', self._is_url_column, self.url_pattern),
            ('identifier', self._is_identifier_column, None),
            ('date_string', self._is_date_string_column, None),
            ('categorical_encoded', self._is_categorical_encoded, None),
            ('binary_flag', self._is_binary_flag, None),
        ]
        
        best_match = ('unknown', 0.0)
        
        for type_name, check_func, pattern in type_checks:
            confidence = check_func(str_series, pattern)
            if confidence > best_match[1]:
                best_match = (type_name, confidence)
        
        analysis['semantic_type'] = best_match[0]
        analysis['confidence'] = best_match[1]
        
        # Add characteristics
        analysis['characteristics'] = {
            'unique_ratio': series.nunique() / len(series),
            'most_common_value': str_series.mode().iloc[0] if len(str_series.mode()) > 0 else None,
            'avg_length': str_series.str.len().mean(),
            'has_numeric_pattern': str_series.str.contains(r'\d').sum() / len(str_series) > 0.5,
        }
        
        return analysis
    
    def _is_email_column(self, series: pd.Series, pattern: str) -> float:
        """Check if column contains email addresses."""
        import re
        if len(series) == 0:
            return 0.0
        matches = series.str.match(pattern, na=False).sum()
        return matches / len(series)
    
    def _is_phone_column(self, series: pd.Series, pattern: str) -> float:
        """Check if column contains phone numbers."""
        import re
        if len(series) == 0:
            return 0.0
        # Clean phone numbers first
        cleaned = series.str.replace(r'[^\d+]', '', regex=True)
        matches = cleaned.str.match(pattern, na=False).sum()
        return matches / len(series)
    
    def _is_url_column(self, series: pd.Series, pattern: str) -> float:
        """Check if column contains URLs."""
        import re
        if len(series) == 0:
            return 0.0
        matches = series.str.match(pattern, na=False).sum()
        return matches / len(series)
    
    def _is_identifier_column(self, series: pd.Series, pattern=None) -> float:
        """Check if column contains identifier-like data."""
        if len(series) == 0:
            return 0.0
        
        # High uniqueness and often contains numbers/letters
        unique_ratio = series.nunique() / len(series)
        contains_mixed = series.str.contains(r'[a-zA-Z].*\d|\d.*[a-zA-Z]', na=False).sum() / len(series)
        
        # Common ID patterns
        common_id_patterns = [
            r'^[A-Z]{2,4}\d+$',  # Letters followed by numbers
            r'^\d{5,}$',         # Long numbers
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',  # UUID
        ]
        
        pattern_match = 0
        for pattern in common_id_patterns:
            pattern_match = max(pattern_match, series.str.match(pattern, na=False).sum() / len(series))
        
        # Score based on uniqueness and pattern matching
        score = 0.0
        if unique_ratio > 0.9:
            score += 0.4
        if contains_mixed > 0.5:
            score += 0.3
        if pattern_match > 0.5:
            score += 0.5
        
        return min(score, 1.0)
    
    def _is_date_string_column(self, series: pd.Series, pattern=None) -> float:
        """Check if column contains date strings."""
        if len(series) == 0:
            return 0.0
        
        # Try to parse as dates
        try:
            parsed = pd.to_datetime(series, errors='coerce')
            valid_dates = parsed.notna().sum()
            return valid_dates / len(series)
        except Exception:
            return 0.0
    
    def _is_categorical_encoded(self, series: pd.Series, pattern=None) -> float:
        """Check if column contains encoded categorical data."""
        if len(series) == 0:
            return 0.0
        
        unique_count = series.nunique()
        total_count = len(series)
        
        # Low cardinality suggests categorical
        if unique_count / total_count < 0.1 and unique_count > 1:
            # Check if values look like codes
            if series.str.len().std() < 2:  # Consistent length
                return 0.7
            return 0.5
        
        return 0.0
    
    def _is_binary_flag(self, series: pd.Series, pattern=None) -> float:
        """Check if column is a binary flag."""
        if len(series) == 0:
            return 0.0
        
        unique_values = set(series.unique())
        
        # Common binary patterns
        binary_patterns = [
            {'0', '1'},
            {'True', 'False'},
            {'true', 'false'},
            {'Y', 'N'},
            {'Yes', 'No'},
            {'yes', 'no'},
            {'T', 'F'},
        ]
        
        for pattern in binary_patterns:
            if unique_values.issubset(pattern) and len(unique_values) == 2:
                return 0.9
        
        # Numeric 0/1 binary
        if unique_values.issubset({0, 1, '0', '1'}) and len(unique_values) == 2:
            return 0.8
        
        return 0.0