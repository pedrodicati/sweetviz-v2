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

        # Create basic data profile
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

        prompt = f"""
        Analyze this dataset and provide a concise summary:
        
        Dataset shape: {profile['shape']}
        Columns: {profile['columns'][:10]}  # Limit for prompt size
        Data types: {str(profile['dtypes'])[:500]}
        Missing values: {str(profile['missing_values'])[:300]}
        
        Provide a 2-3 sentence summary highlighting the most important characteristics.
        """

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"AI insight generation failed: {str(e)}"

    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using basic statistical methods"""
        # Placeholder implementation - would use AI for more sophisticated analysis
        anomalies = {}

        for col in df.select_dtypes(include="number").columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                anomalies[col] = {
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100,
                    "description": f"Found {len(outliers)} outliers in {col}",
                }

        return anomalies

    def explain_correlations(self, correlations: Dict[str, float]) -> str:
        """Explain correlation patterns"""
        if not correlations:
            return "No significant correlations found."

        strong_corr = {k: v for k, v in correlations.items() if abs(v) > 0.7}
        if strong_corr:
            pairs = list(strong_corr.keys())[:3]  # Top 3
            return f"Strong correlations detected between: {', '.join(pairs)}"
        else:
            return "No strong correlations detected in the data."


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
