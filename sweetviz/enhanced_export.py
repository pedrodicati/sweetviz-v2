"""
Enhanced export functionality for sweetviz v2
"""

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from sweetviz.modern_config import ExportFormat, get_config


class EnhancedExporter:
    """Enhanced export functionality with multiple format support"""

    def __init__(self):
        self.config = get_config()

    def export_report(
        self,
        report_data: Dict[str, Any],
        filepath: Union[str, Path],
        formats: Optional[List[ExportFormat]] = None,
    ) -> Dict[str, str]:
        """Export report in specified formats

        Args:
            report_data: Report data dictionary
            filepath: Base filepath (without extension)
            formats: List of formats to export, defaults to config

        Returns:
            Dictionary mapping format to exported filepath
        """
        if formats is None:
            formats = self.config.export_formats

        results = {}
        base_path = Path(filepath)

        for fmt in formats:
            try:
                if fmt == ExportFormat.HTML:
                    output_path = self._export_html(
                        report_data, base_path.with_suffix(".html")
                    )
                elif fmt == ExportFormat.JSON:
                    output_path = self._export_json(
                        report_data, base_path.with_suffix(".json")
                    )
                elif fmt == ExportFormat.SVG:
                    output_path = self._export_svg_charts(report_data, base_path)
                elif fmt == ExportFormat.PDF:
                    output_path = self._export_pdf(
                        report_data, base_path.with_suffix(".pdf")
                    )
                else:
                    continue

                results[fmt.value] = str(output_path)

            except Exception as e:
                print(f"Warning: Failed to export {fmt.value}: {e}")
                continue

        return results

    def _export_html(self, report_data: Dict[str, Any], filepath: Path) -> Path:
        """Export as enhanced HTML with modern features"""
        html_content = self._generate_enhanced_html(report_data)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filepath

    def _export_json(self, report_data: Dict[str, Any], filepath: Path) -> Path:
        """Export as machine-readable JSON"""
        # Create a clean, structured JSON export
        json_data = {
            "metadata": {
                "sweetviz_version": "2.3.1",
                "export_format": "json",
                "timestamp": pd.Timestamp.now().isoformat(),
                "dataset_info": report_data.get("dataset_info", {}),
            },
            "summary": self._extract_summary_stats(report_data),
            "features": self._extract_feature_data(report_data),
            "associations": self._extract_associations(report_data),
            "charts": self._extract_chart_data(report_data),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, default=str)

        return filepath

    def _export_svg_charts(self, report_data: Dict[str, Any], base_path: Path) -> Path:
        """Export all charts as SVG files"""
        svg_dir = base_path.parent / f"{base_path.name}_charts_svg"
        svg_dir.mkdir(exist_ok=True)

        chart_count = 0
        charts = report_data.get("charts", {})

        for chart_name, chart_data in charts.items():
            try:
                svg_path = svg_dir / f"{chart_name}.svg"

                if chart_data.get("type") == "plotly":
                    self._export_plotly_svg(chart_data, svg_path)
                else:
                    self._export_matplotlib_svg(chart_data, svg_path)

                chart_count += 1

            except Exception as e:
                print(f"Warning: Failed to export chart {chart_name} as SVG: {e}")
                continue

        # Create an index file
        index_path = svg_dir / "index.html"
        self._create_svg_index(svg_dir, index_path)

        return svg_dir

    def _export_pdf(self, report_data: Dict[str, Any], filepath: Path) -> Path:
        """Export as PDF using weasyprint"""
        try:
            import weasyprint

            html_content = self._generate_enhanced_html(report_data, for_pdf=True)

            # Create a temporary HTML file
            temp_html = filepath.with_suffix(".temp.html")
            with open(temp_html, "w", encoding="utf-8") as f:
                f.write(html_content)

            # Convert to PDF
            weasyprint.HTML(filename=str(temp_html)).write_pdf(str(filepath))

            # Clean up temporary file
            temp_html.unlink()

            return filepath

        except ImportError:
            raise ImportError(
                "weasyprint is required for PDF export. Install with: pip install weasyprint"
            )

    def _generate_enhanced_html(
        self, report_data: Dict[str, Any], for_pdf: bool = False
    ) -> str:
        """Generate enhanced HTML with modern features"""
        # This would integrate with the existing HTML generation
        # For now, return a basic structure
        theme_class = self._get_theme_class()

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sweetviz Report</title>
    <link rel="stylesheet" href="sweetviz.css">
    <link rel="stylesheet" href="enhanced_themes.css">
    {"" if for_pdf else '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'}
    <style>
        body {{ 
            margin: 0; 
            padding: 20px; 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
    </style>
</head>
<body class="{theme_class}">
    <div class="container fade-in">
        <div class="summary-header">
            <h1>Data Analysis Report</h1>
            <p>Generated with Sweetviz v2</p>
        </div>
        
        <div class="content">
            {self._render_report_content(report_data)}
        </div>
    </div>
    
    {"" if for_pdf else self._get_interactive_scripts()}
</body>
</html>
        """

        return html_template

    def _get_theme_class(self) -> str:
        """Get CSS class for current theme"""
        from sweetviz.modern_config import Theme

        theme_map = {
            Theme.MODERN_DARK: "theme-modern-dark",
            Theme.MODERN_LIGHT: "theme-modern-light",
            Theme.HIGH_CONTRAST: "theme-high-contrast",
            Theme.DEFAULT: "",
            Theme.CLASSIC: "",
        }

        return theme_map.get(self.config.theme, "")

    def _render_report_content(self, report_data: Dict[str, Any]) -> str:
        """Render the main report content"""
        # This would integrate with existing rendering logic
        content = ""

        # Add summary section
        if "summary" in report_data:
            content += self._render_summary_section(report_data["summary"])

        # Add feature sections
        if "features" in report_data:
            content += self._render_features_section(report_data["features"])

        # Add associations
        if "associations" in report_data:
            content += self._render_associations_section(report_data["associations"])

        return content

    def _render_summary_section(self, summary_data: Dict[str, Any]) -> str:
        """Render dataset summary section"""
        return f"""
        <div class="summary-section slide-in">
            <h2>Dataset Overview</h2>
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-label">Rows:</span>
                    <span class="stat-value">{summary_data.get('num_rows', 'N/A')}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Columns:</span>
                    <span class="stat-value">{summary_data.get('num_columns', 'N/A')}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Missing Values:</span>
                    <span class="stat-value missing-value">{summary_data.get('missing_values', 'N/A')}</span>
                </div>
            </div>
        </div>
        """

    def _render_features_section(self, features_data: Dict[str, Any]) -> str:
        """Render features section"""
        content = '<div class="features-section">'
        content += "<h2>Feature Analysis</h2>"

        for feature_name, feature_data in features_data.items():
            content += f"""
            <div class="feature-detail slide-in">
                <h3>{feature_name}</h3>
                <div class="chart-container">
                    {self._render_feature_chart(feature_data)}
                </div>
            </div>
            """

        content += "</div>"
        return content

    def _render_feature_chart(self, feature_data: Dict[str, Any]) -> str:
        """Render individual feature chart"""
        if "chart" in feature_data:
            chart_data = feature_data["chart"]
            if chart_data.get("type") == "plotly":
                return chart_data.get("html", "")
            else:
                img_data = chart_data.get("image_base64", "")
                return f'<img src="data:image/png;base64,{img_data}" alt="Chart" class="chart-image">'
        return "<p>No chart available</p>"

    def _render_associations_section(self, associations_data: Dict[str, Any]) -> str:
        """Render associations section"""
        return f"""
        <div class="associations-section slide-in">
            <h2>Feature Associations</h2>
            <div class="chart-container">
                {self._render_associations_chart(associations_data)}
            </div>
        </div>
        """

    def _render_associations_chart(self, associations_data: Dict[str, Any]) -> str:
        """Render associations chart"""
        if "chart" in associations_data:
            chart_data = associations_data["chart"]
            if chart_data.get("type") == "plotly":
                return chart_data.get("html", "")
            else:
                img_data = chart_data.get("image_base64", "")
                return f'<img src="data:image/png;base64,{img_data}" alt="Associations" class="chart-image">'
        return "<p>No associations chart available</p>"

    def _get_interactive_scripts(self) -> str:
        """Get JavaScript for interactive features"""
        return """
        <script>
        // Add interactivity for enhanced features
        document.addEventListener('DOMContentLoaded', function() {
            // Add chart controls
            const charts = document.querySelectorAll('.chart-container');
            charts.forEach(chart => {
                chart.classList.add('interactive-chart');
                
                const controls = document.createElement('div');
                controls.className = 'chart-controls';
                controls.innerHTML = `
                    <button class="chart-control-btn" onclick="downloadChart(this)" title="Download">
                        📥
                    </button>
                    <button class="chart-control-btn" onclick="fullscreenChart(this)" title="Fullscreen">
                        🔍
                    </button>
                `;
                chart.appendChild(controls);
            });
            
            // Add loading states
            const images = document.querySelectorAll('.chart-image');
            images.forEach(img => {
                img.addEventListener('load', function() {
                    this.style.opacity = '1';
                });
                img.style.opacity = '0';
                img.style.transition = 'opacity 0.3s ease';
            });
        });
        
        function downloadChart(button) {
            // Implementation would depend on chart type
            alert('Download functionality would be implemented here');
        }
        
        function fullscreenChart(button) {
            const chart = button.closest('.chart-container');
            if (chart.requestFullscreen) {
                chart.requestFullscreen();
            }
        }
        </script>
        """

    def _extract_summary_stats(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary statistics for JSON export"""
        summary = report_data.get("summary", {})
        return {
            "num_rows": summary.get("num_rows"),
            "num_columns": summary.get("num_columns"),
            "missing_values": summary.get("missing_values"),
            "duplicate_rows": summary.get("duplicate_rows"),
            "data_types": summary.get("data_types", {}),
        }

    def _extract_feature_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract feature analysis data for JSON export"""
        features = {}
        for feature_name, feature_data in report_data.get("features", {}).items():
            features[feature_name] = {
                "type": feature_data.get("type"),
                "statistics": feature_data.get("statistics", {}),
                "missing_count": feature_data.get("missing_count"),
                "unique_count": feature_data.get("unique_count"),
                "chart_data": feature_data.get("chart", {}).get("data", {}),
            }
        return features

    def _extract_associations(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract association data for JSON export"""
        return report_data.get("associations", {})

    def _extract_chart_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all chart data for JSON export"""
        charts = {}

        # Extract feature charts
        for feature_name, feature_data in report_data.get("features", {}).items():
            if "chart" in feature_data:
                charts[f"feature_{feature_name}"] = feature_data["chart"].get(
                    "data", {}
                )

        # Extract association charts
        if "associations" in report_data and "chart" in report_data["associations"]:
            charts["associations"] = report_data["associations"]["chart"].get(
                "data", {}
            )

        return charts

    def _export_plotly_svg(self, chart_data: Dict[str, Any], filepath: Path) -> None:
        """Export Plotly chart as SVG"""
        try:
            import plotly.io as pio

            fig = chart_data["figure"]
            pio.write_image(fig, str(filepath), format="svg", engine="kaleido")

        except ImportError:
            raise ImportError(
                "kaleido is required for SVG export. Install with: pip install kaleido"
            )

    def _export_matplotlib_svg(
        self, chart_data: Dict[str, Any], filepath: Path
    ) -> None:
        """Export matplotlib chart as SVG"""
        # This would require recreating the matplotlib figure
        # For now, just create a placeholder
        with open(filepath, "w") as f:
            f.write("<svg></svg>")

    def _create_svg_index(self, svg_dir: Path, index_path: Path) -> None:
        """Create an index HTML file for SVG exports"""
        svg_files = list(svg_dir.glob("*.svg"))

        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Chart Gallery</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .chart-item { border: 1px solid #ddd; padding: 10px; border-radius: 8px; }
        .chart-item h3 { margin-top: 0; }
        .chart-item svg, .chart-item img { width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>Chart Gallery</h1>
    <div class="chart-grid">
        """

        for svg_file in svg_files:
            chart_name = svg_file.stem.replace("_", " ").title()
            html_content += f"""
            <div class="chart-item">
                <h3>{chart_name}</h3>
                <object data="{svg_file.name}" type="image/svg+xml" width="100%" height="300">
                    <img src="{svg_file.name}" alt="{chart_name}">
                </object>
            </div>
            """

        html_content += """
    </div>
</body>
</html>
        """

        with open(index_path, "w") as f:
            f.write(html_content)


# Global instance
_enhanced_exporter = EnhancedExporter()


def get_enhanced_exporter() -> EnhancedExporter:
    """Get the global enhanced exporter instance"""
    return _enhanced_exporter
