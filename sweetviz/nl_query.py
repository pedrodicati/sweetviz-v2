"""
Natural Language Query Interface for sweetviz v2 - Phase 5
Allows users to ask questions about their data in natural language
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import pandas as pd
import numpy as np

from sweetviz.modern_config import get_config, LLMProvider


class QueryParser(ABC):
    """Abstract base class for natural language query parsing"""

    @abstractmethod
    def parse_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse natural language query and return structured information"""
        pass

    @abstractmethod
    def execute_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute the query and return results"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this parser is available"""
        pass


class RuleBasedQueryParser(QueryParser):
    """Rule-based natural language query parser"""

    def __init__(self):
        # Define query patterns and their corresponding operations
        self.query_patterns = {
            # Statistics queries
            r'(mean|average) of (.+)': self._get_mean,
            r'(median) of (.+)': self._get_median,
            r'(std|standard deviation) of (.+)': self._get_std,
            r'(min|minimum) of (.+)': self._get_min,
            r'(max|maximum) of (.+)': self._get_max,
            r'(sum|total) of (.+)': self._get_sum,
            r'(count|number) of (.+)': self._get_count,
            
            # Distribution queries
            r'distribution of (.+)': self._get_distribution,
            r'unique values in (.+)': self._get_unique_values,
            r'value counts for (.+)': self._get_value_counts,
            
            # Missing data queries
            r'missing (values|data) in (.+)': self._get_missing_info,
            r'null (values|data) in (.+)': self._get_missing_info,
            r'how many missing (.+)': self._get_missing_count,
            
            # Correlation queries
            r'correlation between (.+) and (.+)': self._get_correlation,
            r'correlate (.+) with (.+)': self._get_correlation,
            r'relationship between (.+) and (.+)': self._get_correlation,
            
            # Data info queries
            r'shape of (dataset|data)': self._get_shape,
            r'size of (dataset|data)': self._get_shape,
            r'columns in (dataset|data)': self._get_columns,
            r'data types': self._get_dtypes,
            r'info about (.+)': self._get_column_info,
            
            # Filtering/selection queries
            r'show me (.+) where (.+)': self._filter_data,
            r'filter (.+) by (.+)': self._filter_data,
            r'rows where (.+)': self._filter_rows,
            
            # Grouping queries
            r'group (.+) by (.+)': self._group_by,
            r'(.+) grouped by (.+)': self._group_by,
            
            # Top/bottom queries
            r'top (\d+) (.+)': self._get_top_n,
            r'bottom (\d+) (.+)': self._get_bottom_n,
            r'highest (.+)': self._get_highest,
            r'lowest (.+)': self._get_lowest,
        }

    def is_available(self) -> bool:
        """Rule-based parser is always available"""
        return True

    def parse_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse natural language query using rule-based patterns"""
        query = query.lower().strip()
        
        for pattern, handler in self.query_patterns.items():
            match = re.search(pattern, query)
            if match:
                return {
                    "pattern": pattern,
                    "matches": match.groups(),
                    "handler": handler.__name__,
                    "query_type": self._classify_query_type(pattern)
                }
        
        return {
            "pattern": None,
            "matches": [],
            "handler": None,
            "query_type": "unknown",
            "error": "Could not understand the query"
        }

    def execute_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute the natural language query"""
        parsed = self.parse_query(query, df)
        
        if parsed.get("error"):
            return {"error": parsed["error"], "suggestions": self._get_suggestions()}
        
        try:
            handler_name = parsed["handler"]
            handler = getattr(self, handler_name)
            matches = parsed["matches"]
            
            result = handler(df, *matches)
            
            return {
                "query": query,
                "query_type": parsed["query_type"],
                "result": result,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": f"Error executing query: {str(e)}",
                "query": query,
                "status": "error"
            }

    def _classify_query_type(self, pattern: str) -> str:
        """Classify the type of query based on pattern"""
        if any(word in pattern for word in ['mean', 'median', 'std', 'min', 'max', 'sum']):
            return "statistics"
        elif any(word in pattern for word in ['distribution', 'unique', 'count']):
            return "distribution"
        elif any(word in pattern for word in ['missing', 'null']):
            return "missing_data"
        elif any(word in pattern for word in ['correlation', 'relationship']):
            return "correlation"
        elif any(word in pattern for word in ['shape', 'size', 'columns', 'types']):
            return "data_info"
        elif any(word in pattern for word in ['filter', 'where', 'show']):
            return "filtering"
        elif any(word in pattern for word in ['group']):
            return "grouping"
        elif any(word in pattern for word in ['top', 'bottom', 'highest', 'lowest']):
            return "ranking"
        else:
            return "other"

    def _find_column(self, df: pd.DataFrame, column_name: str) -> Optional[str]:
        """Find the best matching column name"""
        column_name = column_name.strip()
        
        # Exact match
        if column_name in df.columns:
            return column_name
        
        # Case-insensitive match
        for col in df.columns:
            if col.lower() == column_name.lower():
                return col
        
        # Partial match
        for col in df.columns:
            if column_name.lower() in col.lower() or col.lower() in column_name.lower():
                return col
        
        return None

    # Statistical operations
    def _get_mean(self, df: pd.DataFrame, stat_type: str, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"Column '{col}' is not numeric"}
        
        return {"mean": float(df[col].mean()), "column": col}

    def _get_median(self, df: pd.DataFrame, stat_type: str, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"Column '{col}' is not numeric"}
        
        return {"median": float(df[col].median()), "column": col}

    def _get_std(self, df: pd.DataFrame, stat_type: str, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"Column '{col}' is not numeric"}
        
        return {"std": float(df[col].std()), "column": col}

    def _get_min(self, df: pd.DataFrame, stat_type: str, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        return {"min": df[col].min(), "column": col}

    def _get_max(self, df: pd.DataFrame, stat_type: str, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        return {"max": df[col].max(), "column": col}

    def _get_sum(self, df: pd.DataFrame, stat_type: str, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"Column '{col}' is not numeric"}
        
        return {"sum": float(df[col].sum()), "column": col}

    def _get_count(self, df: pd.DataFrame, stat_type: str, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        return {"count": int(df[col].count()), "non_null_count": int(df[col].count()), "total_rows": len(df), "column": col}

    # Distribution operations
    def _get_distribution(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        if pd.api.types.is_numeric_dtype(df[col]):
            return {
                "column": col,
                "type": "numeric",
                "describe": df[col].describe().to_dict(),
                "quartiles": {
                    "Q1": float(df[col].quantile(0.25)),
                    "Q2": float(df[col].quantile(0.5)),
                    "Q3": float(df[col].quantile(0.75))
                }
            }
        else:
            return {
                "column": col,
                "type": "categorical",
                "value_counts": df[col].value_counts().head(10).to_dict(),
                "unique_count": int(df[col].nunique())
            }

    def _get_unique_values(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        unique_vals = df[col].unique()
        return {
            "column": col,
            "unique_count": len(unique_vals),
            "unique_values": unique_vals[:20].tolist()  # Limit to first 20
        }

    def _get_value_counts(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        return {
            "column": col,
            "value_counts": df[col].value_counts().head(10).to_dict()
        }

    # Missing data operations
    def _get_missing_info(self, df: pd.DataFrame, data_type: str, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        missing_count = df[col].isnull().sum()
        return {
            "column": col,
            "missing_count": int(missing_count),
            "missing_percentage": float(missing_count / len(df) * 100),
            "total_rows": len(df)
        }

    def _get_missing_count(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        return self._get_missing_info(df, "values", column)

    # Correlation operations
    def _get_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        column1 = self._find_column(df, col1)
        column2 = self._find_column(df, col2)
        
        if not column1:
            return {"error": f"Column '{col1}' not found"}
        if not column2:
            return {"error": f"Column '{col2}' not found"}
        
        if not pd.api.types.is_numeric_dtype(df[column1]):
            return {"error": f"Column '{column1}' is not numeric"}
        if not pd.api.types.is_numeric_dtype(df[column2]):
            return {"error": f"Column '{column2}' is not numeric"}
        
        correlation = df[column1].corr(df[column2])
        
        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr > 0.8:
            strength = "very strong"
        elif abs_corr > 0.6:
            strength = "strong"
        elif abs_corr > 0.4:
            strength = "moderate"
        elif abs_corr > 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "positive" if correlation > 0 else "negative"
        
        return {
            "column1": column1,
            "column2": column2,
            "correlation": float(correlation),
            "strength": strength,
            "direction": direction,
            "interpretation": f"{strength} {direction} correlation"
        }

    # Data info operations
    def _get_shape(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "shape": df.shape
        }

    def _get_columns(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        return {
            "columns": df.columns.tolist(),
            "column_count": len(df.columns)
        }

    def _get_dtypes(self, df: pd.DataFrame) -> Dict[str, Any]:
        dtype_counts = df.dtypes.value_counts().to_dict()
        return {
            "data_types": df.dtypes.to_dict(),
            "type_summary": {str(k): int(v) for k, v in dtype_counts.items()}
        }

    def _get_column_info(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        info = {
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique())
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            info.update({
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            })
        
        return info

    # Placeholder methods for more complex operations
    def _filter_data(self, df: pd.DataFrame, *args) -> Dict[str, Any]:
        return {"error": "Complex filtering not yet implemented in rule-based parser"}

    def _filter_rows(self, df: pd.DataFrame, condition: str) -> Dict[str, Any]:
        return {"error": "Row filtering not yet implemented in rule-based parser"}

    def _group_by(self, df: pd.DataFrame, *args) -> Dict[str, Any]:
        return {"error": "Group by operations not yet implemented in rule-based parser"}

    def _get_top_n(self, df: pd.DataFrame, n: str, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        try:
            n_int = int(n)
            top_values = df.nlargest(n_int, col)[col].tolist()
            return {"column": col, "top_n": n_int, "values": top_values}
        except:
            return {"error": f"Invalid number: {n}"}

    def _get_bottom_n(self, df: pd.DataFrame, n: str, column: str) -> Dict[str, Any]:
        col = self._find_column(df, column)
        if not col:
            return {"error": f"Column '{column}' not found"}
        
        try:
            n_int = int(n)
            bottom_values = df.nsmallest(n_int, col)[col].tolist()
            return {"column": col, "bottom_n": n_int, "values": bottom_values}
        except:
            return {"error": f"Invalid number: {n}"}

    def _get_highest(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        return self._get_max(df, "max", column)

    def _get_lowest(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        return self._get_min(df, "min", column)

    def _get_suggestions(self) -> List[str]:
        """Get query suggestions for users"""
        return [
            "mean of [column_name]",
            "distribution of [column_name]",
            "missing values in [column_name]",
            "correlation between [col1] and [col2]",
            "shape of dataset",
            "columns in data",
            "unique values in [column_name]",
            "top 5 [column_name]",
            "info about [column_name]"
        ]


class AIQueryParser(QueryParser):
    """AI-powered natural language query parser using LLMs"""

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
        """Check if AI parser is available"""
        try:
            import openai
            return self.api_key is not None
        except ImportError:
            return False

    def parse_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse query using AI to understand intent and extract parameters"""
        if not self.is_available():
            return {"error": "AI query parser not available (OpenAI not configured)"}

        # Create context about the dataset
        context = f"""
        Dataset Information:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}
        - Data types: {df.dtypes.value_counts().to_dict()}
        """

        prompt = f"""
        Given this dataset context and user query, extract the intent and parameters:
        
        {context}
        
        User Query: "{query}"
        
        Respond with JSON containing:
        {{
            "intent": "one of: statistics, distribution, missing_data, correlation, data_info, filtering, grouping, ranking",
            "operation": "specific operation like mean, median, correlation, etc.",
            "columns": ["list", "of", "column", "names"],
            "parameters": {{"any": "additional parameters"}},
            "executable": true/false
        }}
        """

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )
            
            import json
            result = json.loads(response.choices[0].message.content.strip())
            return result
            
        except Exception as e:
            return {"error": f"AI parsing failed: {str(e)}"}

    def execute_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute query using AI assistance"""
        parsed = self.parse_query(query, df)
        
        if parsed.get("error"):
            return parsed

        # Fallback to rule-based execution for now
        # In a full implementation, this would use AI to generate and execute pandas code
        rule_parser = RuleBasedQueryParser()
        return rule_parser.execute_query(query, df)


class NaturalLanguageQueryInterface:
    """Main interface for natural language queries"""

    def __init__(self):
        self.parsers = {
            "rule_based": RuleBasedQueryParser(),
            "ai_powered": None
        }
        self._initialize_ai_parser()

    def _initialize_ai_parser(self):
        """Initialize AI parser if configured"""
        config = get_config()
        if hasattr(config, 'ai_features') and config.ai_features.enabled:
            if config.ai_features.llm_provider == LLMProvider.OPENAI:
                self.parsers["ai_powered"] = AIQueryParser(
                    api_key=config.ai_features.api_key,
                    model=config.ai_features.model_name or "gpt-3.5-turbo"
                )

    def query(self, query: str, df: pd.DataFrame, use_ai: bool = False) -> Dict[str, Any]:
        """
        Execute a natural language query on the dataset.
        
        Args:
            query: Natural language question about the data
            df: pandas DataFrame to query
            use_ai: Whether to use AI-powered parsing (requires OpenAI configuration)
            
        Returns:
            Dictionary with query results or error information
        """
        if use_ai and self.parsers["ai_powered"] and self.parsers["ai_powered"].is_available():
            return self.parsers["ai_powered"].execute_query(query, df)
        else:
            return self.parsers["rule_based"].execute_query(query, df)

    def get_suggestions(self) -> List[str]:
        """Get example queries that can be executed"""
        return self.parsers["rule_based"]._get_suggestions()

    def is_ai_available(self) -> bool:
        """Check if AI-powered querying is available"""
        return (self.parsers["ai_powered"] is not None and 
                self.parsers["ai_powered"].is_available())


# Global query interface instance
_nl_query_interface = NaturalLanguageQueryInterface()


def ask_question(query: str, df: pd.DataFrame, use_ai: bool = False) -> Dict[str, Any]:
    """
    Ask a natural language question about the data.
    
    Args:
        query: Natural language question (e.g., "What is the mean of age?")
        df: pandas DataFrame to analyze
        use_ai: Whether to use AI-powered parsing
        
    Returns:
        Dictionary with answer or error information
    """
    return _nl_query_interface.query(query, df, use_ai)


def get_query_suggestions() -> List[str]:
    """Get example queries that can be asked"""
    return _nl_query_interface.get_suggestions()


def get_nl_query_interface() -> NaturalLanguageQueryInterface:
    """Get the global natural language query interface"""
    return _nl_query_interface