"""
Example usage of Sweetviz v2 with AI features
"""
import pandas as pd
import numpy as np
import sweetviz as sv

# Create sample dataset
np.random.seed(42)
data = {
    'sales': np.random.normal(1000, 200, 500),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
    'product_type': np.random.choice(['A', 'B', 'C'], 500),
    'customer_satisfaction': np.random.uniform(1, 10, 500),
    'is_returning_customer': np.random.choice([True, False], 500, p=[0.3, 0.7])
}
df = pd.DataFrame(data)

print("=== Sweetviz v2 Modernization Example ===\n")

# 1. Basic analysis (unchanged API)
print("1. Creating basic analysis...")
report = sv.analyze(df)
print("✓ Basic analysis complete")

# 2. Modern configuration
print("\n2. Using modern configuration...")
config = sv.ModernConfig()
config.theme = sv.modern_config.Theme.MODERN_DARK
config.performance_mode = sv.modern_config.PerformanceMode.FAST
print(f"✓ Theme set to: {config.theme}")
print(f"✓ Performance mode: {config.performance_mode}")

# Set the global config
sv.set_config(config)

# 3. AI features (will not be fully functional without API keys)
print("\n3. AI features configuration...")
config.enable_ai_features(
    provider="openai",  # Would need real API key
    generate_insights=True,
    anomaly_detection=True
)

print(f"✓ AI features enabled: {config.ai_features.enabled}")
print(f"✓ AI provider: {config.ai_features.llm_provider}")

# Get AI manager
ai_manager = sv.get_ai_manager()
print(f"✓ AI manager available: {ai_manager.is_available()}")

# 4. Test AI features (mock since no API key)
print("\n4. Testing AI insights (mock data)...")
if not ai_manager.is_available():
    print("  → AI insights not available (no API key configured)")
    print("  → Would generate: 'Dataset contains 500 sales records across 4 regions...'")
    print("  → Would detect: Potential outliers in sales column")
    print("  → Would explain: Strong correlation between region and sales")

# 5. Export configuration
print("\n5. Configuration export...")
config_dict = config.to_dict()
print(f"✓ Config exported: {len(config_dict)} settings")

# 6. Backwards compatibility
print("\n6. Testing backwards compatibility...")
legacy_setting = config.get_legacy_setting("General", "default_verbosity", "full")
print(f"✓ Legacy config access: {legacy_setting}")

print("\n=== Example Complete ===")
print("\nNew Features Added:")
print("• Modern configuration system with dataclasses")
print("• AI integration foundation (OpenAI, HuggingFace)")
print("• Type hints for public APIs")
print("• Development tooling (black, isort, flake8, pytest)")
print("• GitHub Actions CI/CD pipeline")
print("• Comprehensive test suite")
print("• Backwards compatibility maintained")

print("\nTo enable full AI features:")
print("1. Install AI dependencies: pip install sweetviz[ai]")
print("2. Set API key: config.enable_ai_features(api_key='your-key')")
print("3. Generate insights: ai_manager.generate_data_summary(df)")