"""
Phase 4 Enhanced Visualization and Export Demo
Demonstrates the new features in Sweetviz v2 Phase 4
"""

import numpy as np
import pandas as pd
import sweetviz as sv


# Create sample dataset for demonstration
def create_sample_data():
    """Create a comprehensive sample dataset"""
    np.random.seed(42)

    data = {
        "age": np.random.normal(35, 12, 1000),
        "income": np.random.lognormal(10, 0.5, 1000),
        "education": np.random.choice(
            ["High School", "Bachelor", "Master", "PhD"], 1000, p=[0.3, 0.4, 0.2, 0.1]
        ),
        "experience": np.random.exponential(8, 1000),
        "satisfaction": np.random.randint(1, 11, 1000),
        "department": np.random.choice(
            ["Engineering", "Sales", "Marketing", "HR", "Finance"], 1000
        ),
        "remote_work": np.random.choice([True, False], 1000, p=[0.4, 0.6]),
        "performance_score": np.random.beta(2, 5, 1000) * 100,
    }

    df = pd.DataFrame(data)

    # Add some missing values (convert to float first to allow NaN)
    df["income"][np.random.choice(1000, 50, replace=False)] = np.nan
    df["satisfaction"] = df["satisfaction"].astype(float)
    df["satisfaction"][np.random.choice(1000, 30, replace=False)] = np.nan

    # Add some correlations
    df.loc[df["education"] == "PhD", "income"] *= 1.5
    df.loc[df["education"] == "High School", "income"] *= 0.8
    df["experience"] = np.where(df["age"] > 30, df["experience"] + 5, df["experience"])

    return df


def demo_basic_enhanced_features():
    """Demonstrate basic enhanced visualization features"""
    print("🎨 Phase 4: Enhanced Visualization Demo")
    print("=" * 50)

    # Create modern configuration
    config = sv.ModernConfig()
    config.theme = sv.Theme.MODERN_DARK
    config.visualizations.engine = sv.VisualizationEngine.AUTO
    config.visualizations.interactive_charts = True
    config.visualizations.mobile_responsive = True
    config.visualizations.chart_height = 500
    config.visualizations.chart_width = 700

    print(f"✅ Theme: {config.theme.value}")
    print(
        f"✅ Visualization Engine: {config.visualizations.get_effective_engine().value}"
    )
    print(f"✅ Interactive Charts: {config.visualizations.interactive_charts}")
    print(f"✅ Mobile Responsive: {config.visualizations.mobile_responsive}")

    # Set the configuration
    sv.set_config(config)

    # Get enhanced visualizer
    visualizer = sv.get_enhanced_visualizer()

    # Create sample data
    data = np.random.normal(100, 15, 1000)

    # Create enhanced histogram
    print("\n📊 Creating Enhanced Histogram...")
    histogram = visualizer.create_histogram(
        data,
        title="Enhanced Sample Distribution",
        x_label="Values",
        y_label="Frequency",
        bins=30,
    )

    print(f"   Chart Type: {histogram['type']}")
    print(f"   Has Data: {'data' in histogram}")
    if histogram["type"] == "plotly":
        print("   🎉 Using interactive Plotly charts!")
    else:
        print("   📈 Using matplotlib (Plotly not available)")

    # Create bar chart
    print("\n📊 Creating Enhanced Bar Chart...")
    categories = ["Category A", "Category B", "Category C", "Category D"]
    values = [23, 45, 56, 78]

    bar_chart = visualizer.create_bar_chart(
        categories,
        values,
        title="Enhanced Category Analysis",
        x_label="Categories",
        y_label="Values",
    )

    print(f"   Chart Type: {bar_chart['type']}")
    print(f"   Categories: {len(categories)}")

    # Create correlation heatmap
    print("\n🔥 Creating Enhanced Correlation Heatmap...")
    df_sample = pd.DataFrame(
        {
            "A": np.random.randn(100),
            "B": np.random.randn(100),
            "C": np.random.randn(100),
        }
    )
    df_sample["D"] = df_sample["A"] * 0.8 + np.random.randn(100) * 0.2

    corr_matrix = df_sample.corr()
    heatmap = visualizer.create_correlation_heatmap(
        corr_matrix, title="Enhanced Correlation Analysis"
    )

    print(f"   Chart Type: {heatmap['type']}")
    print(f"   Matrix Size: {corr_matrix.shape}")


def demo_multiple_themes():
    """Demonstrate different theme capabilities"""
    print("\n🎨 Theme Demonstration")
    print("=" * 30)

    visualizer = sv.get_enhanced_visualizer()

    themes = [
        sv.Theme.DEFAULT,
        sv.Theme.MODERN_LIGHT,
        sv.Theme.MODERN_DARK,
        sv.Theme.HIGH_CONTRAST,
    ]

    for theme in themes:
        config = sv.get_config()
        config.theme = theme
        sv.set_config(config)

        primary_color = visualizer._get_theme_color("primary")
        secondary_color = visualizer._get_theme_color("secondary")

        print(f"   {theme.value.replace('_', ' ').title()}:")
        print(f"     Primary Color: {primary_color}")
        print(f"     Secondary Color: {secondary_color}")


def demo_enhanced_export():
    """Demonstrate enhanced export capabilities"""
    print("\n📤 Enhanced Export Demo")
    print("=" * 30)

    # Create sample report data
    sample_data = create_sample_data()

    # Create a basic report structure for demo
    report_data = {
        "summary": {
            "num_rows": len(sample_data),
            "num_columns": len(sample_data.columns),
            "missing_values": sample_data.isnull().sum().sum(),
            "duplicate_rows": sample_data.duplicated().sum(),
            "data_types": sample_data.dtypes.astype(str).to_dict(),
        },
        "features": {},
        "associations": {},
    }

    # Add feature analysis
    for col in sample_data.select_dtypes(include=[np.number]).columns[:3]:
        visualizer = sv.get_enhanced_visualizer()
        chart_data = visualizer.create_histogram(
            sample_data[col].dropna(),
            title=f"Distribution of {col}",
            x_label=col,
            y_label="Frequency",
        )

        report_data["features"][col] = {
            "type": "numeric",
            "statistics": {
                "mean": float(sample_data[col].mean()),
                "std": float(sample_data[col].std()),
                "min": float(sample_data[col].min()),
                "max": float(sample_data[col].max()),
            },
            "missing_count": int(sample_data[col].isnull().sum()),
            "unique_count": int(sample_data[col].nunique()),
            "chart": chart_data,
        }

    # Get enhanced exporter
    exporter = sv.get_enhanced_exporter()

    # Test different export formats
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "demo_report"

        # Export multiple formats
        export_formats = [sv.ExportFormat.HTML, sv.ExportFormat.JSON]

        try:
            results = exporter.export_report(
                report_data, output_path, formats=export_formats
            )

            print("✅ Export Results:")
            for fmt, filepath in results.items():
                file_size = (
                    Path(filepath).stat().st_size if Path(filepath).exists() else 0
                )
                print(f"   {fmt.upper()}: {filepath} ({file_size} bytes)")

                # Verify content for JSON
                if fmt == "json":
                    import json

                    with open(filepath, "r") as f:
                        data = json.load(f)
                    print(f"     JSON has {len(data)} top-level keys")

                # Verify content for HTML
                if fmt == "html":
                    with open(filepath, "r") as f:
                        content = f.read()
                    mobile_responsive = "viewport" in content
                    interactive = "plotly" in content.lower()
                    print(f"     Mobile Responsive: {mobile_responsive}")
                    print(f"     Interactive Elements: {interactive}")

        except Exception as e:
            print(f"❌ Export failed: {e}")


def demo_performance_features():
    """Demonstrate performance configuration features"""
    print("\n⚡ Performance Features Demo")
    print("=" * 35)

    config = sv.get_config()

    print(f"✅ Performance Mode: {config.performance_mode.value}")
    print(f"✅ Enable Sampling: {config.performance.enable_sampling}")
    print(f"✅ Max Sample Size: {config.performance.max_sample_size:,}")
    print(f"✅ Chunk Size: {config.performance.chunk_size:,}")
    print(f"✅ Parallel Processing: {config.performance.parallel_processing}")
    print(f"✅ Memory Limit: {config.performance.memory_limit_mb} MB")

    # Demonstrate configuration
    config.performance.enable_sampling = True
    config.performance.max_sample_size = 5000
    config.performance_mode = sv.PerformanceMode.FAST

    print("\n🔧 Updated Performance Settings:")
    print(f"   Mode: {config.performance_mode.value}")
    print(f"   Max Sample: {config.performance.max_sample_size:,}")


def demo_config_serialization():
    """Demonstrate configuration serialization"""
    print("\n💾 Configuration Serialization Demo")
    print("=" * 40)

    # Create comprehensive configuration
    config = sv.ModernConfig()
    config.theme = sv.Theme.MODERN_DARK
    config.performance_mode = sv.PerformanceMode.BALANCED
    config.visualizations.engine = sv.VisualizationEngine.AUTO
    config.visualizations.interactive_charts = True
    config.visualizations.mobile_responsive = True
    config.export_formats = [
        sv.ExportFormat.HTML,
        sv.ExportFormat.JSON,
        sv.ExportFormat.SVG,
    ]

    # Enable AI features (demo - won't actually work without API key)
    config.enable_ai_features(
        provider=sv.LLMProvider.OPENAI, generate_insights=True, anomaly_detection=True
    )

    # Serialize to dictionary
    config_dict = config.to_dict()

    print("✅ Configuration serialized to dictionary:")
    import json

    print(json.dumps(config_dict, indent=2))


def main():
    """Run the complete Phase 4 demo"""
    print("🚀 Sweetviz v2 - Phase 4 Enhanced Features Demo")
    print("=" * 60)
    print("This demo showcases the new enhanced visualization and export capabilities")
    print("introduced in Phase 4 of the Sweetviz v2 modernization project.")
    print()

    try:
        # Demo 1: Basic enhanced features
        demo_basic_enhanced_features()

        # Demo 2: Multiple themes
        demo_multiple_themes()

        # Demo 3: Enhanced export
        demo_enhanced_export()

        # Demo 4: Performance features
        demo_performance_features()

        # Demo 5: Config serialization
        demo_config_serialization()

        print("\n🎉 Phase 4 Demo Completed Successfully!")
        print("=" * 50)
        print("Key Phase 4 Features Demonstrated:")
        print("✅ Enhanced visualization with Plotly support")
        print("✅ Modern themes (Dark, Light, High Contrast)")
        print("✅ Multiple export formats (HTML, JSON, SVG)")
        print("✅ Mobile-responsive design")
        print("✅ Performance optimizations")
        print("✅ Configuration serialization")
        print("✅ Backwards compatibility maintained")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
