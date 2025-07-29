"""
Modern configuration system for sweetviz v2 with AI feature support
"""

import configparser
import importlib.resources
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Theme(Enum):
    """Available themes for sweetviz reports"""

    DEFAULT = "default"
    MODERN_DARK = "modern_dark"
    MODERN_LIGHT = "modern_light"
    CLASSIC = "classic"


class PerformanceMode(Enum):
    """Performance modes for large dataset processing"""

    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"


class LLMProvider(Enum):
    """Supported LLM providers for AI features"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class AIConfig:
    """Configuration for AI-powered features"""

    enabled: bool = False
    llm_provider: LLMProvider = LLMProvider.OPENAI
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.1
    generate_insights: bool = True
    anomaly_detection: bool = True
    smart_correlations: bool = True

    def is_available(self) -> bool:
        """Check if AI features can be used"""
        if not self.enabled:
            return False
        # Check if required dependencies are available
        try:
            if self.llm_provider == LLMProvider.OPENAI:
                import openai

                return self.api_key is not None
            elif self.llm_provider == LLMProvider.HUGGINGFACE:
                import transformers

                return True
            return False
        except ImportError:
            return False


@dataclass
class ModernConfig:
    """Modern configuration class for sweetviz v2"""

    theme: Theme = Theme.DEFAULT
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    ai_features: AIConfig = field(default_factory=AIConfig)
    export_formats: List[str] = field(default_factory=lambda: ["html"])

    # Legacy compatibility
    _legacy_config: Optional[configparser.ConfigParser] = None

    def __post_init__(self):
        """Initialize legacy config for backwards compatibility"""
        self._legacy_config = configparser.ConfigParser()
        with importlib.resources.open_text("sweetviz", "sweetviz_defaults.ini") as f:
            self._legacy_config.read_file(f)

    def get_legacy_setting(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get setting from legacy config for backwards compatibility"""
        if self._legacy_config:
            return self._legacy_config.get(section, key, fallback=fallback)
        return fallback

    def enable_ai_features(
        self,
        provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Enable AI features with specified configuration"""
        if isinstance(provider, str):
            provider = LLMProvider(provider)

        self.ai_features.enabled = True
        self.ai_features.llm_provider = provider
        if api_key:
            self.ai_features.api_key = api_key

        for key, value in kwargs.items():
            if hasattr(self.ai_features, key):
                setattr(self.ai_features, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "theme": self.theme.value,
            "performance_mode": self.performance_mode.value,
            "ai_features": {
                "enabled": self.ai_features.enabled,
                "llm_provider": self.ai_features.llm_provider.value,
                "model_name": self.ai_features.model_name,
                "max_tokens": self.ai_features.max_tokens,
                "temperature": self.ai_features.temperature,
                "generate_insights": self.ai_features.generate_insights,
                "anomaly_detection": self.ai_features.anomaly_detection,
                "smart_correlations": self.ai_features.smart_correlations,
            },
            "export_formats": self.export_formats,
        }


# Global config instance
_global_config = ModernConfig()


def get_config() -> ModernConfig:
    """Get the global configuration instance"""
    return _global_config


def set_config(config: ModernConfig) -> None:
    """Set the global configuration instance"""
    global _global_config
    _global_config = config


# Backwards compatibility: expose legacy config
config = _global_config._legacy_config
