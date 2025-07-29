"""
Tests for Phase 4 enhanced visualization and export features
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import sweetviz as sv
from sweetviz.enhanced_export import EnhancedExporter
from sweetviz.enhanced_viz import EnhancedVisualizer
from sweetviz.modern_config import (
    ExportFormat,
    Theme,
    VisualizationConfig,
    VisualizationEngine,
)


class TestEnhancedVisualization:
    """Test enhanced visualization features"""

    def setup_method(self):
        """Setup test data"""
        self.test_data = pd.DataFrame(
            {
                "numeric": np.random.randn(100),
                "categorical": np.random.choice(["A", "B", "C"], 100),
                "values": np.random.randint(1, 100, 100),
            }
        )

    def test_enhanced_visualizer_initialization(self):
        """Test enhanced visualizer can be initialized"""
        visualizer = sv.get_enhanced_visualizer()
        assert isinstance(visualizer, EnhancedVisualizer)

    def test_visualization_engine_detection(self):
        """Test visualization engine detection"""
        visualizer = sv.get_enhanced_visualizer()
        engine = visualizer.get_visualization_engine()
        assert engine in [VisualizationEngine.MATPLOTLIB, VisualizationEngine.PLOTLY]

    def test_histogram_creation_matplotlib(self):
        """Test histogram creation with matplotlib"""
        visualizer = sv.get_enhanced_visualizer()

        # Force matplotlib engine
        visualizer.config.visualizations.engine = VisualizationEngine.MATPLOTLIB

        result = visualizer.create_histogram(
            self.test_data["numeric"],
            title="Test Histogram",
            x_label="Values",
            y_label="Frequency",
        )

        assert result["type"] == "matplotlib"
        assert "image_base64" in result
        assert "data" in result

    @patch("sweetviz.enhanced_viz.EnhancedVisualizer._check_plotly_availability")
    def test_histogram_creation_plotly(self, mock_plotly_check):
        """Test histogram creation with plotly when available"""
        mock_plotly_check.return_value = True

        # Skip if plotly not available for testing
        try:
            import plotly.graph_objects as go

            plotly_available = True
        except ImportError:
            plotly_available = False

        if not plotly_available:
            pytest.skip("Plotly not available for testing")

        with patch("plotly.graph_objects.Figure") as mock_fig:
            mock_instance = MagicMock()
            mock_fig.return_value = mock_instance
            mock_instance.to_html.return_value = "<div>Plotly Chart</div>"
            mock_instance.to_json.return_value = '{"data": []}'

            visualizer = sv.get_enhanced_visualizer()
            visualizer._plotly_available = True
            visualizer.config.visualizations.engine = VisualizationEngine.PLOTLY

            result = visualizer.create_histogram(
                self.test_data["numeric"], title="Test Histogram"
            )

            assert result["type"] == "plotly"
            assert "html" in result
            assert "json" in result

    def test_bar_chart_creation(self):
        """Test bar chart creation"""
        visualizer = sv.get_enhanced_visualizer()

        categories = ["A", "B", "C"]
        values = [10, 20, 15]

        result = visualizer.create_bar_chart(
            categories,
            values,
            title="Test Bar Chart",
            x_label="Categories",
            y_label="Values",
        )

        assert result["type"] in ["matplotlib", "plotly"]
        assert "data" in result
        assert result["data"]["categories"] == categories
        assert result["data"]["values"] == values

    def test_correlation_heatmap_creation(self):
        """Test correlation heatmap creation"""
        visualizer = sv.get_enhanced_visualizer()

        # Create a simple correlation matrix
        corr_matrix = self.test_data[["numeric", "values"]].corr()

        result = visualizer.create_correlation_heatmap(
            corr_matrix, title="Test Correlation Heatmap"
        )

        assert result["type"] in ["matplotlib", "plotly"]
        assert "data" in result

    def test_theme_color_selection(self):
        """Test theme-based color selection"""
        visualizer = sv.get_enhanced_visualizer()

        # Test different themes
        visualizer.config.theme = Theme.MODERN_DARK
        color_dark = visualizer._get_theme_color("primary")

        visualizer.config.theme = Theme.MODERN_LIGHT
        color_light = visualizer._get_theme_color("primary")

        assert color_dark != color_light
        assert color_dark.startswith("#")
        assert color_light.startswith("#")

    def test_plotly_template_selection(self):
        """Test plotly template selection based on theme"""
        visualizer = sv.get_enhanced_visualizer()

        visualizer.config.theme = Theme.MODERN_DARK
        template_dark = visualizer._get_plotly_template()

        visualizer.config.theme = Theme.MODERN_LIGHT
        template_light = visualizer._get_plotly_template()

        assert template_dark in ["plotly_dark", "plotly", "plotly_white"]
        assert template_light in ["plotly_dark", "plotly", "plotly_white"]

    def test_visualization_config(self):
        """Test visualization configuration"""
        config = sv.get_config()

        # Test default values
        assert isinstance(config.visualizations, VisualizationConfig)
        assert config.visualizations.interactive_charts is True
        assert config.visualizations.mobile_responsive is True

        # Test configuration changes
        config.visualizations.chart_height = 500
        config.visualizations.chart_width = 800

        assert config.visualizations.chart_height == 500
        assert config.visualizations.chart_width == 800


class TestEnhancedExport:
    """Test enhanced export features"""

    def setup_method(self):
        """Setup test data"""
        self.test_report_data = {
            "summary": {
                "num_rows": 100,
                "num_columns": 3,
                "missing_values": 5,
                "duplicate_rows": 0,
            },
            "features": {
                "numeric_feature": {
                    "type": "numeric",
                    "statistics": {"mean": 10.5, "std": 2.3},
                    "missing_count": 2,
                    "unique_count": 95,
                    "chart": {
                        "type": "matplotlib",
                        "image_base64": "test_image_data",
                        "data": {"x": [1, 2, 3], "bins": 10, "counts": [5, 10, 8]},
                    },
                }
            },
            "associations": {
                "chart": {
                    "type": "matplotlib",
                    "image_base64": "test_heatmap_data",
                    "data": {"corr_matrix": [[1.0, 0.5], [0.5, 1.0]]},
                }
            },
        }

    def test_enhanced_exporter_initialization(self):
        """Test enhanced exporter can be initialized"""
        exporter = sv.get_enhanced_exporter()
        assert isinstance(exporter, EnhancedExporter)

    def test_json_export(self):
        """Test JSON export functionality"""
        exporter = sv.get_enhanced_exporter()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report"

            results = exporter.export_report(
                self.test_report_data, output_path, formats=[ExportFormat.JSON]
            )

            assert "json" in results
            json_file = Path(results["json"])
            assert json_file.exists()

            # Verify JSON content
            with open(json_file, "r") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "summary" in data
            assert "features" in data
            assert data["summary"]["num_rows"] == 100

    def test_html_export(self):
        """Test HTML export functionality"""
        exporter = sv.get_enhanced_exporter()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report"

            results = exporter.export_report(
                self.test_report_data, output_path, formats=[ExportFormat.HTML]
            )

            assert "html" in results
            html_file = Path(results["html"])
            assert html_file.exists()

            # Verify HTML content
            with open(html_file, "r") as f:
                content = f.read()

            assert "<!DOCTYPE html>" in content
            assert "Sweetviz Report" in content
            assert "viewport" in content  # Mobile responsive

    def test_multiple_format_export(self):
        """Test exporting multiple formats"""
        exporter = sv.get_enhanced_exporter()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report"

            results = exporter.export_report(
                self.test_report_data,
                output_path,
                formats=[ExportFormat.HTML, ExportFormat.JSON],
            )

            assert "html" in results
            assert "json" in results
            assert Path(results["html"]).exists()
            assert Path(results["json"]).exists()

    def test_theme_class_generation(self):
        """Test CSS theme class generation"""
        exporter = sv.get_enhanced_exporter()

        # Test different themes
        original_theme = exporter.config.theme

        exporter.config.theme = Theme.MODERN_DARK
        assert "modern-dark" in exporter._get_theme_class()

        exporter.config.theme = Theme.MODERN_LIGHT
        assert "modern-light" in exporter._get_theme_class()

        exporter.config.theme = Theme.HIGH_CONTRAST
        assert "high-contrast" in exporter._get_theme_class()

        # Restore original theme
        exporter.config.theme = original_theme

    def test_summary_stats_extraction(self):
        """Test extraction of summary statistics"""
        exporter = sv.get_enhanced_exporter()

        summary = exporter._extract_summary_stats(self.test_report_data)

        assert summary["num_rows"] == 100
        assert summary["num_columns"] == 3
        assert summary["missing_values"] == 5

    def test_feature_data_extraction(self):
        """Test extraction of feature data"""
        exporter = sv.get_enhanced_exporter()

        features = exporter._extract_feature_data(self.test_report_data)

        assert "numeric_feature" in features
        feature = features["numeric_feature"]
        assert feature["type"] == "numeric"
        assert feature["missing_count"] == 2
        assert feature["statistics"]["mean"] == 10.5

    def test_chart_data_extraction(self):
        """Test extraction of chart data"""
        exporter = sv.get_enhanced_exporter()

        charts = exporter._extract_chart_data(self.test_report_data)

        assert "feature_numeric_feature" in charts
        assert "associations" in charts


class TestPhase4Integration:
    """Test integration of Phase 4 features with existing API"""

    def test_enhanced_config_integration(self):
        """Test that enhanced configuration works with existing API"""
        # Create modern config with enhanced features
        config = sv.ModernConfig()
        config.theme = sv.Theme.MODERN_DARK
        config.visualizations.engine = sv.VisualizationEngine.AUTO
        config.visualizations.interactive_charts = True
        config.export_formats = [sv.ExportFormat.HTML, sv.ExportFormat.JSON]

        sv.set_config(config)

        # Verify config is set
        current_config = sv.get_config()
        assert current_config.theme == sv.Theme.MODERN_DARK
        assert current_config.visualizations.interactive_charts is True
        assert sv.ExportFormat.JSON in current_config.export_formats

    def test_enhanced_features_availability(self):
        """Test that enhanced features are available through API"""
        # Test enhanced visualizer
        visualizer = sv.get_enhanced_visualizer()
        assert hasattr(visualizer, "create_histogram")
        assert hasattr(visualizer, "create_bar_chart")
        assert hasattr(visualizer, "create_correlation_heatmap")

        # Test enhanced exporter
        exporter = sv.get_enhanced_exporter()
        assert hasattr(exporter, "export_report")

    def test_backwards_compatibility(self):
        """Test that Phase 4 doesn't break existing functionality"""
        # Create simple test data
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["x", "y", "z", "x", "y"]})

        # Test that basic analyze still works
        try:
            report = sv.analyze(df)
            assert hasattr(report, "show_html")
        except Exception as e:
            pytest.fail(f"Basic analyze functionality broken: {e}")

    def test_enum_exports(self):
        """Test that enums are properly exported"""
        # Test that all enums are available
        assert hasattr(sv, "Theme")
        assert hasattr(sv, "VisualizationEngine")
        assert hasattr(sv, "ExportFormat")
        assert hasattr(sv, "PerformanceMode")
        assert hasattr(sv, "LLMProvider")

        # Test enum values
        assert sv.Theme.MODERN_DARK.value == "modern_dark"
        assert sv.VisualizationEngine.PLOTLY.value == "plotly"
        assert sv.ExportFormat.JSON.value == "json"

    def test_config_serialization_with_enums(self):
        """Test config serialization works with new enum fields"""
        config = sv.ModernConfig()
        config.theme = sv.Theme.MODERN_DARK
        config.visualizations.engine = sv.VisualizationEngine.PLOTLY
        config.export_formats = [sv.ExportFormat.HTML, sv.ExportFormat.SVG]

        # Test serialization
        config_dict = config.to_dict()

        assert config_dict["theme"] == "modern_dark"
        assert config_dict["visualizations"]["engine"] == "plotly"
        assert "html" in config_dict["export_formats"]
        assert "svg" in config_dict["export_formats"]


if __name__ == "__main__":
    pytest.main([__file__])
