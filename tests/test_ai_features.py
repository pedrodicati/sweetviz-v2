"""
Tests for AI integration features
"""
import pytest
import pandas as pd
import sweetviz as sv
from sweetviz.modern_config import ModernConfig, LLMProvider, Theme, PerformanceMode
from sweetviz.ai_insights import AIInsightManager, OpenAIInsightProvider, HuggingFaceInsightProvider


@pytest.fixture
def sample_df():
    """Sample dataframe for testing"""
    return pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5, 100],  # Has outlier
        'categorical': ['A', 'B', 'A', 'B', 'A', 'C'],
        'text': ['hello', 'world', 'test', 'data', 'sample', 'text']
    })


class TestModernConfig:
    """Test modern configuration system"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ModernConfig()
        assert config.theme == Theme.DEFAULT
        assert config.performance_mode == PerformanceMode.BALANCED
        assert not config.ai_features.enabled
        assert config.export_formats == ["html"]
    
    def test_enable_ai_features(self):
        """Test enabling AI features"""
        config = ModernConfig()
        config.enable_ai_features(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model_name="gpt-4"
        )
        
        assert config.ai_features.enabled
        assert config.ai_features.llm_provider == LLMProvider.OPENAI
        assert config.ai_features.api_key == "test-key"
        assert config.ai_features.model_name == "gpt-4"
    
    def test_config_serialization(self):
        """Test config serialization to dict"""
        config = ModernConfig()
        config.enable_ai_features(provider="openai")
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "theme" in config_dict
        assert "ai_features" in config_dict
        assert config_dict["ai_features"]["enabled"] is True
    
    def test_global_config(self):
        """Test global config management"""
        original_config = sv.get_config()
        
        new_config = ModernConfig()
        new_config.theme = Theme.MODERN_DARK
        sv.set_config(new_config)
        
        retrieved_config = sv.get_config()
        assert retrieved_config.theme == Theme.MODERN_DARK
        
        # Restore original config
        sv.set_config(original_config)


class TestAIInsights:
    """Test AI insight functionality"""
    
    def test_ai_manager_creation(self):
        """Test AI manager creation"""
        manager = sv.get_ai_manager()
        assert isinstance(manager, AIInsightManager)
    
    def test_ai_not_available_by_default(self):
        """Test that AI is not available by default"""
        manager = sv.get_ai_manager()
        assert not manager.is_available()
    
    def test_openai_provider_without_key(self, sample_df):
        """Test OpenAI provider without API key"""
        provider = OpenAIInsightProvider()
        assert not provider.is_available()
        
        summary = provider.generate_data_summary(sample_df)
        assert "not available" in summary.lower()
    
    def test_huggingface_provider_availability(self):
        """Test HuggingFace provider availability check"""
        provider = HuggingFaceInsightProvider()
        # This will be False in CI environment without transformers
        availability = provider.is_available()
        assert isinstance(availability, bool)
    
    def test_anomaly_detection_basic(self, sample_df):
        """Test basic anomaly detection"""
        provider = OpenAIInsightProvider()
        anomalies = provider.detect_anomalies(sample_df)
        
        # Should detect the outlier in 'numeric' column
        assert isinstance(anomalies, dict)
        if 'numeric' in anomalies:
            assert anomalies['numeric']['count'] > 0


class TestAIIntegrationWithSweetViz:
    """Test AI integration with main sweetviz functionality"""
    
    def test_analyze_with_ai_config(self, sample_df):
        """Test analyze function with AI configuration"""
        # This should work even without AI enabled
        report = sv.analyze(sample_df)
        assert report is not None
    
    def test_config_backwards_compatibility(self):
        """Test that new config doesn't break legacy config access"""
        config = sv.get_config()
        
        # Should still be able to access legacy config
        legacy_setting = config.get_legacy_setting("General", "default_verbosity", "full")
        assert isinstance(legacy_setting, str)
    
    def test_new_api_availability(self):
        """Test that new API functions are available"""
        assert hasattr(sv, 'ModernConfig')
        assert hasattr(sv, 'get_config')
        assert hasattr(sv, 'set_config')
        assert hasattr(sv, 'get_ai_manager')


class TestAIFeatureConfiguration:
    """Test AI feature configuration options"""
    
    @pytest.mark.parametrize("provider", [LLMProvider.OPENAI, LLMProvider.HUGGINGFACE])
    def test_different_providers(self, provider):
        """Test configuration with different AI providers"""
        config = ModernConfig()
        config.enable_ai_features(provider=provider)
        
        assert config.ai_features.llm_provider == provider
        assert config.ai_features.enabled
    
    def test_ai_config_parameters(self):
        """Test various AI configuration parameters"""
        config = ModernConfig()
        config.enable_ai_features(
            provider=LLMProvider.OPENAI,
            max_tokens=500,
            temperature=0.5,
            generate_insights=False
        )
        
        assert config.ai_features.max_tokens == 500
        assert config.ai_features.temperature == 0.5
        assert not config.ai_features.generate_insights