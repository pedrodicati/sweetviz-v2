"""
AI integration foundation for sweetviz v2
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from sweetviz.modern_config import LLMProvider, get_config


class AIInsightProvider(ABC):
    """Abstract base class for AI insight providers"""

    @abstractmethod
    def generate_data_summary(self, df: pd.DataFrame) -> str:
        """Generate natural language summary of the dataset"""
        pass

    @abstractmethod
    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and explain anomalies in the data"""
        pass

    @abstractmethod
    def explain_correlations(self, correlations: Dict[str, float]) -> str:
        """Provide insights about feature correlations"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available"""
        pass


class OpenAIInsightProvider(AIInsightProvider):
    """OpenAI-powered insight provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI provider is available"""
        try:
            import openai

            return self.api_key is not None
        except ImportError:
            return False

    def generate_data_summary(self, df: pd.DataFrame) -> str:
        """Generate natural language summary using OpenAI"""
        if not self.is_available():
            return "AI insights not available (OpenAI not configured)"

        # Create comprehensive data profile
        profile = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": (
                df.describe().to_dict()
                if len(df.select_dtypes(include="number").columns) > 0
                else {}
            ),
        }

        # Add more sophisticated analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Calculate data quality metrics
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        duplicate_rows = df.duplicated().sum()
        
        # Detect potential issues
        issues = []
        if missing_percentage > 10:
            issues.append(f"High missing data ({missing_percentage:.1f}%)")
        if duplicate_rows > 0:
            issues.append(f"{duplicate_rows} duplicate rows")
        
        # Check for potential outliers in numeric columns
        outlier_cols = []
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                outlier_cols.append(f"{col} ({len(outliers)} outliers)")

        if outlier_cols:
            issues.append(f"Potential outliers in: {', '.join(outlier_cols)}")

        prompt = f"""
        Analyze this dataset and provide a comprehensive summary for a data scientist:
        
        **Dataset Overview:**
        - Shape: {profile['shape'][0]:,} rows × {profile['shape'][1]} columns
        - Data Types: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical
        - Missing Data: {missing_percentage:.1f}% overall
        - Duplicates: {duplicate_rows} rows
        
        **Key Columns:** {', '.join(profile['columns'][:8])}{'...' if len(profile['columns']) > 8 else ''}
        
        **Data Quality Issues:** {'; '.join(issues) if issues else 'None detected'}
        
        **Numeric Features Summary:** {str(profile['numeric_summary'])[:400]}...
        
        Please provide:
        1. A 2-3 sentence overview of what this dataset represents
        2. Key insights about data quality and structure
        3. Recommendations for next steps in analysis
        
        Be concise but insightful, focusing on actionable insights for EDA.
        """

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"AI insight generation failed: {str(e)}"

    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using statistical and AI-enhanced methods"""
        anomalies = {
            "statistical_outliers": {},
            "pattern_anomalies": {},
            "ai_insights": [],
            "recommendations": []
        }

        # Enhanced statistical outlier detection
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() < 10:  # Skip columns with too few values
                continue
                
            col_data = df[col].dropna()
            
            # Multiple outlier detection methods
            # 1. IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            # 2. Z-score method
            mean_val = col_data.mean()
            std_val = col_data.std()
            z_scores = abs((col_data - mean_val) / std_val)
            zscore_outliers = df[abs((df[col] - mean_val) / std_val) > 3]
            
            # 3. Modified Z-score using median
            median_val = col_data.median()
            mad = (col_data - median_val).abs().median()
            modified_z_scores = 0.6745 * (col_data - median_val) / mad
            mad_outliers = df[abs(0.6745 * (df[col] - median_val) / mad) > 3.5] if mad > 0 else pd.DataFrame()
            
            if len(iqr_outliers) > 0 or len(zscore_outliers) > 0:
                anomalies["statistical_outliers"][col] = {
                    "iqr_outliers": {
                        "count": len(iqr_outliers),
                        "percentage": len(iqr_outliers) / len(df) * 100,
                        "bounds": [float(lower_bound), float(upper_bound)],
                        "values": iqr_outliers[col].tolist()[:10]  # Sample of outlier values
                    },
                    "zscore_outliers": {
                        "count": len(zscore_outliers),
                        "percentage": len(zscore_outliers) / len(df) * 100,
                        "threshold": 3.0
                    },
                    "mad_outliers": {
                        "count": len(mad_outliers),
                        "percentage": len(mad_outliers) / len(df) * 100,
                        "threshold": 3.5
                    }
                }

        # Pattern-based anomaly detection
        # 1. Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            anomalies["pattern_anomalies"]["constant_columns"] = {
                "columns": constant_cols,
                "description": "Columns with no variation in values"
            }

        # 2. Check for highly missing columns
        high_missing_cols = []
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct > 80:
                high_missing_cols.append({"column": col, "missing_percentage": missing_pct})
        
        if high_missing_cols:
            anomalies["pattern_anomalies"]["high_missing_columns"] = {
                "columns": high_missing_cols,
                "description": "Columns with >80% missing values"
            }

        # 3. Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            anomalies["pattern_anomalies"]["duplicate_rows"] = {
                "count": int(duplicate_count),
                "percentage": duplicate_count / len(df) * 100,
                "description": f"Found {duplicate_count} duplicate rows"
            }

        # 4. Check for unusual data types or formats
        unusual_patterns = []
        for col in df.select_dtypes(include=['object']).columns:
            # Check for mixed data types in string columns
            sample_values = df[col].dropna().head(100)
            if len(sample_values) > 0:
                # Check if numeric values are stored as strings
                numeric_like = 0
                for val in sample_values:
                    try:
                        float(str(val))
                        numeric_like += 1
                    except:
                        pass
                
                if numeric_like / len(sample_values) > 0.8:
                    unusual_patterns.append({
                        "column": col,
                        "issue": "numeric_as_string",
                        "description": f"Column '{col}' contains mostly numeric values stored as strings"
                    })

        if unusual_patterns:
            anomalies["pattern_anomalies"]["data_type_issues"] = unusual_patterns

        # Generate AI insights if OpenAI is available
        if self.is_available():
            ai_insights = []
            
            # Summarize findings for AI analysis
            summary = f"""
            Anomaly Detection Results:
            - Statistical outliers found in {len(anomalies['statistical_outliers'])} columns
            - Pattern anomalies: {len(anomalies['pattern_anomalies'])} types detected
            - Duplicate rows: {duplicate_count}
            - Constant columns: {len(constant_cols)}
            - High missing data columns: {len(high_missing_cols)}
            """
            
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user", 
                        "content": f"Analyze these data anomalies and provide insights: {summary}"
                    }],
                    max_tokens=150,
                    temperature=0.2,
                )
                ai_insights.append(response.choices[0].message.content.strip())
            except Exception as e:
                ai_insights.append(f"AI analysis failed: {str(e)}")
            
            anomalies["ai_insights"] = ai_insights

        # Generate recommendations
        recommendations = []
        
        if len(anomalies["statistical_outliers"]) > 0:
            recommendations.append("Review statistical outliers: they may be data errors or genuinely unusual observations")
        
        if duplicate_count > 0:
            recommendations.append("Consider removing duplicate rows or investigating why they exist")
        
        if constant_cols:
            recommendations.append("Remove constant columns as they provide no information for analysis")
        
        if high_missing_cols:
            recommendations.append("Consider dropping columns with >80% missing data or investigate data collection issues")
        
        if unusual_patterns:
            recommendations.append("Fix data type issues for better analysis accuracy")

        anomalies["recommendations"] = recommendations

        return anomalies

    def explain_correlations(self, correlations: Dict[str, float]) -> str:
        """Explain correlation patterns using AI insights"""
        if not correlations:
            return "No significant correlations found in the dataset."

        # Categorize correlations by strength
        strong_positive = {k: v for k, v in correlations.items() if v > 0.7}
        strong_negative = {k: v for k, v in correlations.items() if v < -0.7}
        moderate_positive = {k: v for k, v in correlations.items() if 0.3 < v <= 0.7}
        moderate_negative = {k: v for k, v in correlations.items() if -0.7 <= v < -0.3}
        weak_correlations = {k: v for k, v in correlations.items() if -0.3 <= v <= 0.3}

        # Generate detailed analysis
        analysis_parts = []

        if strong_positive:
            top_positive = sorted(strong_positive.items(), key=lambda x: x[1], reverse=True)[:3]
            analysis_parts.append(
                f"**Strong Positive Correlations** (>0.7): {len(strong_positive)} found. "
                f"Top relationships: {', '.join([f'{pair} ({corr:.2f})' for pair, corr in top_positive])}"
            )

        if strong_negative:
            top_negative = sorted(strong_negative.items(), key=lambda x: x[1])[:3]
            analysis_parts.append(
                f"**Strong Negative Correlations** (<-0.7): {len(strong_negative)} found. "
                f"Top relationships: {', '.join([f'{pair} ({corr:.2f})' for pair, corr in top_negative])}"
            )

        if moderate_positive:
            analysis_parts.append(f"**Moderate Positive Correlations**: {len(moderate_positive)} relationships (0.3-0.7)")

        if moderate_negative:
            analysis_parts.append(f"**Moderate Negative Correlations**: {len(moderate_negative)} relationships (-0.7--0.3)")

        # Generate AI insights if available
        if self.is_available() and (strong_positive or strong_negative):
            try:
                # Prepare correlation summary for AI
                strong_corrs = {**strong_positive, **strong_negative}
                top_correlations = sorted(strong_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                
                prompt = f"""
                Analyze these feature correlations and provide insights:
                
                Strong correlations found:
                {chr(10).join([f"- {pair}: {corr:.3f}" for pair, corr in top_correlations])}
                
                Total correlations: {len(correlations)} pairs analyzed
                Strong correlations: {len(strong_positive)} positive, {len(strong_negative)} negative
                Moderate correlations: {len(moderate_positive)} positive, {len(moderate_negative)} negative
                
                Provide:
                1. Interpretation of the strongest relationships
                2. Potential implications for modeling/analysis
                3. Any concerns about multicollinearity
                
                Be concise and actionable.
                """

                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250,
                    temperature=0.2,
                )
                ai_insight = response.choices[0].message.content.strip()
                analysis_parts.append(f"\n**AI Analysis:** {ai_insight}")
                
            except Exception as e:
                analysis_parts.append(f"\nAI analysis failed: {str(e)}")

        # Add recommendations
        recommendations = []
        
        if len(strong_positive) > 3:
            recommendations.append("High multicollinearity detected - consider feature selection or dimensionality reduction")
        
        if len(strong_negative) > 0:
            recommendations.append("Strong negative correlations found - useful for feature engineering")
        
        if len(weak_correlations) / len(correlations) > 0.8:
            recommendations.append("Most features show weak correlations - consider feature creation or domain knowledge")

        if recommendations:
            analysis_parts.append(f"\n**Recommendations:** {'; '.join(recommendations)}")

        return " ".join(analysis_parts) if analysis_parts else "No significant correlation patterns detected."


class HuggingFaceInsightProvider(AIInsightProvider):
    """HuggingFace-powered insight provider"""

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy initialization of HuggingFace pipeline"""
        if self._pipeline is None:
            try:
                from transformers import pipeline

                self._pipeline = pipeline("summarization", model=self.model_name)
            except ImportError:
                raise ImportError(
                    "Transformers package not installed. Install with: pip install transformers"
                )
        return self._pipeline

    def is_available(self) -> bool:
        """Check if HuggingFace provider is available"""
        try:
            import transformers

            return True
        except ImportError:
            return False

    def generate_data_summary(self, df: pd.DataFrame) -> str:
        """Generate summary using HuggingFace models"""
        if not self.is_available():
            return "AI insights not available (HuggingFace not configured)"

        # Create a structured description of the data
        text = f"Dataset with {df.shape[0]} rows and {df.shape[1]} columns. "
        text += f"Column types: {df.dtypes.value_counts().to_dict()}. "
        text += f"Missing values: {df.isnull().sum().sum()} total."

        # For now, return the structured description
        # In a full implementation, would use summarization models
        return text

    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic anomaly detection"""
        return {}  # Placeholder

    def explain_correlations(self, correlations: Dict[str, float]) -> str:
        """Basic correlation explanation"""
        return "Correlation analysis available."  # Placeholder


class AIInsightManager:
    """Manager for AI insight providers"""

    def __init__(self):
        self._provider: Optional[AIInsightProvider] = None
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize AI provider based on configuration"""
        config = get_config()
        if not config.ai_features.enabled or not config.ai_features.is_available():
            return

        if config.ai_features.llm_provider == LLMProvider.OPENAI:
            self._provider = OpenAIInsightProvider(
                api_key=config.ai_features.api_key,
                model=config.ai_features.model_name or "gpt-3.5-turbo",
            )
        elif config.ai_features.llm_provider == LLMProvider.HUGGINGFACE:
            self._provider = HuggingFaceInsightProvider(
                model_name=config.ai_features.model_name or "facebook/bart-large-cnn"
            )

    def is_available(self) -> bool:
        """Check if AI insights are available"""
        return self._provider is not None and self._provider.is_available()

    def generate_data_summary(self, df: pd.DataFrame) -> Optional[str]:
        """Generate AI-powered data summary"""
        if not self.is_available():
            return None
        return self._provider.generate_data_summary(df)

    def detect_anomalies(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect anomalies using AI"""
        if not self.is_available():
            return None
        return self._provider.detect_anomalies(df)

    def explain_correlations(self, correlations: Dict[str, float]) -> Optional[str]:
        """Explain correlations using AI"""
        if not self.is_available():
            return None
        return self._provider.explain_correlations(correlations)


# Global AI manager instance
_ai_manager = AIInsightManager()


def get_ai_manager() -> AIInsightManager:
    """Get the global AI insight manager"""
    return _ai_manager
