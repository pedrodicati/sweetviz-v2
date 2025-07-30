# sweetviz public interface
# -----------------------------------------------------------------------------------
try:
    from importlib.metadata import metadata  # Python 3.8+
except ImportError:
    from importlib_metadata import metadata  # Python 3.7

# These are the main API functions
from sweetviz.ai_insights import get_ai_manager

# Phase 5: MLOps platform integrations
from sweetviz.mlops_integrations import get_mlops_manager

# Phase 5: Natural language query interface
from sweetviz.nl_query import ask_question, get_query_suggestions, get_nl_query_interface

# This is the config_parser, use to customize settings
from sweetviz.config import config as config_parser

# This is the main report class; holds the report data
# and is used to output the final report
from sweetviz.dataframe_report import DataframeReport
from sweetviz.feature_config import FeatureConfig

# Modern configuration and AI features
from sweetviz.modern_config import ModernConfig, get_config, set_config
from sweetviz.sv_public import analyze, compare, compare_intra

# Phase 4: Enhanced visualization and export capabilities
from sweetviz.enhanced_viz import get_enhanced_visualizer
from sweetviz.enhanced_export import get_enhanced_exporter

# Expose modern config enums for convenience
from sweetviz.modern_config import (
    Theme,
    VisualizationEngine,
    ExportFormat,
    PerformanceMode,
    LLMProvider,
)

_metadata = metadata("sweetviz")
__title__ = _metadata["name"]
__version__ = _metadata["version"]
__author__ = _metadata["Author-email"]
__license__ = "MIT"

__all__ = [
    "analyze",
    "compare",
    "compare_intra",
    "FeatureConfig",
    "DataframeReport",
    "config_parser",
    "ModernConfig",
    "get_config",
    "set_config",
    "get_ai_manager",
    "get_enhanced_visualizer",
    "get_enhanced_exporter",
    "get_mlops_manager",
    "ask_question",
    "get_query_suggestions",
    "get_nl_query_interface",
    "Theme",
    "VisualizationEngine", 
    "ExportFormat",
    "PerformanceMode",
    "LLMProvider",
]
