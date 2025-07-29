"""
Enhanced visualization module for sweetviz v2 with Plotly support
"""

import base64
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sweetviz.modern_config import VisualizationEngine, get_config


class EnhancedVisualizer:
    """Enhanced visualization engine with Plotly and matplotlib support"""

    def __init__(self):
        self.config = get_config()
        self._plotly_available = self._check_plotly_availability()

    def _check_plotly_availability(self) -> bool:
        """Check if Plotly is available"""
        try:
            import plotly.graph_objects as go

            return True
        except ImportError:
            return False

    def get_visualization_engine(self) -> VisualizationEngine:
        """Get the effective visualization engine to use"""
        return self.config.visualizations.get_effective_engine()

    def create_histogram(
        self,
        data: Union[pd.Series, np.ndarray, List],
        title: str = "",
        x_label: str = "",
        y_label: str = "Frequency",
        bins: int = 30,
        color: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a histogram using the configured visualization engine"""
        engine = self.get_visualization_engine()

        if engine == VisualizationEngine.PLOTLY and self._plotly_available:
            return self._create_plotly_histogram(
                data, title, x_label, y_label, bins, color, **kwargs
            )
        else:
            return self._create_matplotlib_histogram(
                data, title, x_label, y_label, bins, color, **kwargs
            )

    def create_bar_chart(
        self,
        categories: List[str],
        values: List[float],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        color: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a bar chart using the configured visualization engine"""
        engine = self.get_visualization_engine()

        if engine == VisualizationEngine.PLOTLY and self._plotly_available:
            return self._create_plotly_bar_chart(
                categories, values, title, x_label, y_label, color, **kwargs
            )
        else:
            return self._create_matplotlib_bar_chart(
                categories, values, title, x_label, y_label, color, **kwargs
            )

    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Correlation Heatmap",
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a correlation heatmap"""
        engine = self.get_visualization_engine()

        if engine == VisualizationEngine.PLOTLY and self._plotly_available:
            return self._create_plotly_heatmap(correlation_matrix, title, **kwargs)
        else:
            return self._create_matplotlib_heatmap(correlation_matrix, title, **kwargs)

    def _create_plotly_histogram(
        self,
        data: Union[pd.Series, np.ndarray, List],
        title: str,
        x_label: str,
        y_label: str,
        bins: int,
        color: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create histogram using Plotly"""
        try:
            import plotly.graph_objects as go
            import plotly.io as pio

            # Configure theme
            template = self._get_plotly_template()

            fig = go.Figure(
                data=[
                    go.Histogram(
                        x=data,
                        nbinsx=bins,
                        marker_color=color or self._get_theme_color("primary"),
                        opacity=0.8,
                    )
                ]
            )

            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                template=template,
                height=self.config.visualizations.chart_height,
                width=self.config.visualizations.chart_width,
                showlegend=False,
            )

            # Make responsive if enabled
            if self.config.visualizations.mobile_responsive:
                fig.update_layout(
                    autosize=True,
                    margin=dict(l=50, r=50, t=50, b=50),
                )

            return {
                "type": "plotly",
                "figure": fig,
                "html": fig.to_html(include_plotlyjs="cdn"),
                "json": fig.to_json(),
                "data": {"x": list(data)},
            }

        except ImportError:
            # Fallback to matplotlib
            return self._create_matplotlib_histogram(
                data, title, x_label, y_label, bins, color, **kwargs
            )

    def _create_plotly_bar_chart(
        self,
        categories: List[str],
        values: List[float],
        title: str,
        x_label: str,
        y_label: str,
        color: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create bar chart using Plotly"""
        try:
            import plotly.graph_objects as go

            template = self._get_plotly_template()

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=categories,
                        y=values,
                        marker_color=color or self._get_theme_color("primary"),
                        opacity=0.8,
                    )
                ]
            )

            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                template=template,
                height=self.config.visualizations.chart_height,
                width=self.config.visualizations.chart_width,
                showlegend=False,
            )

            if self.config.visualizations.mobile_responsive:
                fig.update_layout(
                    autosize=True,
                    margin=dict(l=50, r=50, t=50, b=50),
                )

            return {
                "type": "plotly",
                "figure": fig,
                "html": fig.to_html(include_plotlyjs="cdn"),
                "json": fig.to_json(),
                "data": {"categories": categories, "values": values},
            }

        except ImportError:
            return self._create_matplotlib_bar_chart(
                categories, values, title, x_label, y_label, color, **kwargs
            )

    def _create_plotly_heatmap(
        self, correlation_matrix: pd.DataFrame, title: str, **kwargs
    ) -> Dict[str, Any]:
        """Create correlation heatmap using Plotly"""
        try:
            import plotly.graph_objects as go

            template = self._get_plotly_template()

            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale="RdBu",
                    zmid=0,
                    colorbar=dict(title="Correlation"),
                )
            )

            fig.update_layout(
                title=title,
                template=template,
                height=self.config.visualizations.chart_height,
                width=self.config.visualizations.chart_width,
            )

            if self.config.visualizations.mobile_responsive:
                fig.update_layout(autosize=True)

            return {
                "type": "plotly",
                "figure": fig,
                "html": fig.to_html(include_plotlyjs="cdn"),
                "json": fig.to_json(),
                "data": correlation_matrix.to_dict(),
            }

        except ImportError:
            return self._create_matplotlib_heatmap(correlation_matrix, title, **kwargs)

    def _create_matplotlib_histogram(
        self,
        data: Union[pd.Series, np.ndarray, List],
        title: str,
        x_label: str,
        y_label: str,
        bins: int,
        color: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create histogram using matplotlib (fallback)"""
        fig, ax = plt.subplots(
            figsize=(
                self.config.visualizations.chart_width / 100,
                self.config.visualizations.chart_height / 100,
            )
        )

        n, bins_edges, patches = ax.hist(
            data, bins=bins, color=color or self._get_theme_color("primary"), alpha=0.8
        )

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Apply theme styling
        self._apply_matplotlib_theme(fig, ax)

        # Convert to base64
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=150 if self.config.visualizations.high_dpi else 100,
            bbox_inches="tight",
            transparent=True,
        )
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)

        return {
            "type": "matplotlib",
            "image_base64": img_base64,
            "data": {"x": list(data), "bins": bins, "counts": list(n)},
        }

    def _create_matplotlib_bar_chart(
        self,
        categories: List[str],
        values: List[float],
        title: str,
        x_label: str,
        y_label: str,
        color: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create bar chart using matplotlib (fallback)"""
        fig, ax = plt.subplots(
            figsize=(
                self.config.visualizations.chart_width / 100,
                self.config.visualizations.chart_height / 100,
            )
        )

        bars = ax.bar(
            categories,
            values,
            color=color or self._get_theme_color("primary"),
            alpha=0.8,
        )

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Rotate x-axis labels if needed
        if len(categories) > 5:
            plt.xticks(rotation=45, ha="right")

        self._apply_matplotlib_theme(fig, ax)

        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=150 if self.config.visualizations.high_dpi else 100,
            bbox_inches="tight",
            transparent=True,
        )
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)

        return {
            "type": "matplotlib",
            "image_base64": img_base64,
            "data": {"categories": categories, "values": values},
        }

    def _create_matplotlib_heatmap(
        self, correlation_matrix: pd.DataFrame, title: str, **kwargs
    ) -> Dict[str, Any]:
        """Create correlation heatmap using matplotlib (fallback)"""
        fig, ax = plt.subplots(
            figsize=(
                self.config.visualizations.chart_width / 100,
                self.config.visualizations.chart_height / 100,
            )
        )

        im = ax.imshow(correlation_matrix.values, cmap="RdBu", vmin=-1, vmax=1)

        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.index)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(correlation_matrix.index)

        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation")

        self._apply_matplotlib_theme(fig, ax)

        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=150 if self.config.visualizations.high_dpi else 100,
            bbox_inches="tight",
            transparent=True,
        )
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)

        return {
            "type": "matplotlib",
            "image_base64": img_base64,
            "data": correlation_matrix.to_dict(),
        }

    def _get_plotly_template(self) -> str:
        """Get Plotly template based on current theme"""
        from sweetviz.modern_config import Theme

        if self.config.theme == Theme.MODERN_DARK:
            return "plotly_dark"
        elif self.config.theme == Theme.MODERN_LIGHT:
            return "plotly_white"
        elif self.config.theme == Theme.HIGH_CONTRAST:
            return "plotly_dark"
        else:
            return "plotly"

    def _get_theme_color(self, color_type: str) -> str:
        """Get color for current theme"""
        from sweetviz.modern_config import Theme

        colors = {
            Theme.DEFAULT: {"primary": "#1f77b4", "secondary": "#ff7f0e"},
            Theme.MODERN_DARK: {"primary": "#636EFA", "secondary": "#EF553B"},
            Theme.MODERN_LIGHT: {"primary": "#2E86C1", "secondary": "#E67E22"},
            Theme.HIGH_CONTRAST: {"primary": "#FFFFFF", "secondary": "#FFFF00"},
            Theme.CLASSIC: {"primary": "#1f77b4", "secondary": "#ff7f0e"},
        }

        theme_colors = colors.get(self.config.theme, colors[Theme.DEFAULT])
        return theme_colors.get(color_type, theme_colors["primary"])

    def _apply_matplotlib_theme(self, fig, ax) -> None:
        """Apply theme styling to matplotlib figure"""
        from sweetviz.modern_config import Theme

        if self.config.theme == Theme.MODERN_DARK:
            fig.patch.set_facecolor("#2F2F2F")
            ax.set_facecolor("#2F2F2F")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.tick_params(colors="white")
        elif self.config.theme == Theme.HIGH_CONTRAST:
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.tick_params(colors="white")

    def export_svg(self, chart_data: Dict[str, Any], filename: str) -> str:
        """Export chart as SVG"""
        if chart_data["type"] == "plotly":
            try:
                import plotly.io as pio

                fig = chart_data["figure"]
                svg_str = pio.to_image(fig, format="svg", engine="kaleido")
                with open(filename, "wb") as f:
                    f.write(svg_str)
                return filename
            except ImportError:
                pass

        # Fallback to matplotlib SVG
        if chart_data["type"] == "matplotlib":
            # Would need to recreate the figure for SVG export
            # For now, return the filename indicating it's not supported
            return f"{filename}.png_fallback"

        return filename

    def get_chart_json(self, chart_data: Dict[str, Any]) -> str:
        """Get chart data as JSON for machine-readable export"""
        return json.dumps(chart_data.get("data", {}), indent=2)


# Global instance
_enhanced_visualizer = EnhancedVisualizer()


def get_enhanced_visualizer() -> EnhancedVisualizer:
    """Get the global enhanced visualizer instance"""
    return _enhanced_visualizer
