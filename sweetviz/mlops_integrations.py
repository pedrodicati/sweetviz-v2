"""
MLOps platform integrations for sweetviz v2 - Phase 5
Provides integrations with MLflow, Weights & Biases, and other ML platforms
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import json
import pandas as pd
from pathlib import Path

from sweetviz.modern_config import get_config


class MLOpsIntegration(ABC):
    """Abstract base class for MLOps platform integrations"""

    @abstractmethod
    def export_report(
        self, 
        report_data: Dict[str, Any], 
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Export sweetviz report to the MLOps platform"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this integration is available"""
        pass

    @abstractmethod
    def log_dataset_summary(self, df: pd.DataFrame, name: str = "dataset") -> bool:
        """Log basic dataset summary metrics"""
        pass


class MLflowIntegration(MLOpsIntegration):
    """Integration with MLflow for experiment tracking"""

    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri
        self._mlflow = None

    def _get_mlflow(self):
        """Lazy initialization of MLflow"""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
            except ImportError:
                raise ImportError(
                    "MLflow not installed. Install with: pip install mlflow"
                )
        return self._mlflow

    def is_available(self) -> bool:
        """Check if MLflow is available"""
        try:
            import mlflow
            return True
        except ImportError:
            return False

    def export_report(
        self, 
        report_data: Dict[str, Any], 
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Export sweetviz report to MLflow"""
        if not self.is_available():
            return {"error": "MLflow not available"}

        mlflow = self._get_mlflow()
        
        try:
            # Set experiment if provided
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            with mlflow.start_run() as run:
                # Log tags
                if tags:
                    mlflow.set_tags(tags)

                # Log dataset metrics
                if "dataset_info" in report_data:
                    dataset_info = report_data["dataset_info"]
                    mlflow.log_param("dataset_shape", str(dataset_info.get("shape", "unknown")))
                    mlflow.log_param("dataset_columns", dataset_info.get("num_columns", 0))
                    mlflow.log_param("missing_values", dataset_info.get("missing_values", 0))

                # Log summary statistics
                if "summary_stats" in report_data:
                    for key, value in report_data["summary_stats"].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"summary_{key}", value)

                # Log the full report as artifact
                report_json = json.dumps(report_data, indent=2, default=str)
                mlflow.log_text(report_json, "sweetviz_report.json")

                # Log any generated visualizations
                if "charts" in report_data:
                    for chart_name, chart_data in report_data["charts"].items():
                        if isinstance(chart_data, str) and chart_data.endswith(('.png', '.svg')):
                            mlflow.log_artifact(chart_data, f"charts/{chart_name}")

                return {
                    "status": "success",
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "mlflow_ui": mlflow.get_tracking_uri()
                }

        except Exception as e:
            return {"error": f"MLflow export failed: {str(e)}"}

    def log_dataset_summary(self, df: pd.DataFrame, name: str = "dataset") -> bool:
        """Log basic dataset summary to MLflow"""
        if not self.is_available():
            return False

        mlflow = self._get_mlflow()
        
        try:
            with mlflow.start_run():
                # Basic dataset metrics
                mlflow.log_param(f"{name}_shape", str(df.shape))
                mlflow.log_param(f"{name}_columns", df.shape[1])
                mlflow.log_metric(f"{name}_rows", df.shape[0])
                mlflow.log_metric(f"{name}_missing_values", df.isnull().sum().sum())
                
                # Column type distribution
                dtype_counts = df.dtypes.value_counts().to_dict()
                for dtype, count in dtype_counts.items():
                    mlflow.log_metric(f"{name}_columns_{str(dtype)}", count)

                # Memory usage
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                mlflow.log_metric(f"{name}_memory_mb", memory_mb)

            return True
        except Exception:
            return False


class WandBIntegration(MLOpsIntegration):
    """Integration with Weights & Biases for experiment tracking"""

    def __init__(self, project: Optional[str] = None, entity: Optional[str] = None):
        self.project = project
        self.entity = entity
        self._wandb = None

    def _get_wandb(self):
        """Lazy initialization of wandb"""
        if self._wandb is None:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                raise ImportError(
                    "Weights & Biases not installed. Install with: pip install wandb"
                )
        return self._wandb

    def is_available(self) -> bool:
        """Check if wandb is available"""
        try:
            import wandb
            return True
        except ImportError:
            return False

    def export_report(
        self, 
        report_data: Dict[str, Any], 
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Export sweetviz report to Weights & Biases"""
        if not self.is_available():
            return {"error": "Weights & Biases not available"}

        wandb = self._wandb()
        
        try:
            # Initialize run
            run_config = {}
            if tags:
                run_config.update(tags)
            
            run = wandb.init(
                project=self.project or experiment_name or "sweetviz-analysis",
                entity=self.entity,
                name=experiment_name,
                config=run_config
            )

            # Log dataset info as config
            if "dataset_info" in report_data:
                wandb.config.update(report_data["dataset_info"])

            # Log summary statistics as metrics
            if "summary_stats" in report_data:
                wandb.log(report_data["summary_stats"])

            # Create wandb Table for structured data
            if "tables" in report_data:
                for table_name, table_data in report_data["tables"].items():
                    if isinstance(table_data, pd.DataFrame):
                        wandb_table = wandb.Table(dataframe=table_data)
                        wandb.log({f"table_{table_name}": wandb_table})

            # Log the full report as artifact
            report_artifact = wandb.Artifact(
                name="sweetviz_report", 
                type="report",
                description="Sweetviz EDA report"
            )
            
            # Save report JSON to temporary file and add to artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(report_data, f, indent=2, default=str)
                report_artifact.add_file(f.name, name="sweetviz_report.json")

            wandb.log_artifact(report_artifact)

            run_info = {
                "status": "success",
                "run_id": run.id,
                "run_name": run.name,
                "project": run.project,
                "wandb_url": run.url
            }

            wandb.finish()
            return run_info

        except Exception as e:
            return {"error": f"Weights & Biases export failed: {str(e)}"}

    def log_dataset_summary(self, df: pd.DataFrame, name: str = "dataset") -> bool:
        """Log basic dataset summary to Weights & Biases"""
        if not self.is_available():
            return False

        wandb = self._wandb()
        
        try:
            # Initialize a quick run just for logging
            with wandb.init(job_type="data_summary") as run:
                # Basic dataset metrics
                summary_data = {
                    f"{name}_shape": str(df.shape),
                    f"{name}_rows": df.shape[0],
                    f"{name}_columns": df.shape[1],
                    f"{name}_missing_values": int(df.isnull().sum().sum()),
                    f"{name}_memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
                }
                
                # Column type distribution
                dtype_counts = df.dtypes.value_counts().to_dict()
                for dtype, count in dtype_counts.items():
                    summary_data[f"{name}_columns_{str(dtype)}"] = count

                wandb.log(summary_data)

            return True
        except Exception:
            return False


class MLOpsManager:
    """Manager for MLOps platform integrations"""

    def __init__(self):
        self._integrations: Dict[str, MLOpsIntegration] = {}
        self._initialize_integrations()

    def _initialize_integrations(self):
        """Initialize available MLOps integrations"""
        config = get_config()
        
        # Initialize MLflow if configured
        if hasattr(config, 'mlops') and hasattr(config.mlops, 'mlflow_enabled'):
            if config.mlops.mlflow_enabled:
                tracking_uri = getattr(config.mlops, 'mlflow_tracking_uri', None)
                self._integrations['mlflow'] = MLflowIntegration(tracking_uri)

        # Initialize Weights & Biases if configured
        if hasattr(config, 'mlops') and hasattr(config.mlops, 'wandb_enabled'):
            if config.mlops.wandb_enabled:
                project = getattr(config.mlops, 'wandb_project', None)
                entity = getattr(config.mlops, 'wandb_entity', None)
                self._integrations['wandb'] = WandBIntegration(project, entity)

    def add_integration(self, name: str, integration: MLOpsIntegration):
        """Add a custom MLOps integration"""
        self._integrations[name] = integration

    def get_integration(self, name: str) -> Optional[MLOpsIntegration]:
        """Get a specific MLOps integration"""
        return self._integrations.get(name)

    def list_available_integrations(self) -> List[str]:
        """List all available integrations"""
        return [name for name, integration in self._integrations.items() 
                if integration.is_available()]

    def export_to_mlflow(
        self, 
        report_data: Dict[str, Any], 
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Export report to MLflow"""
        integration = self.get_integration('mlflow')
        if integration:
            return integration.export_report(report_data, experiment_name, tags)
        
        # Create temporary integration if not configured
        temp_integration = MLflowIntegration()
        if temp_integration.is_available():
            return temp_integration.export_report(report_data, experiment_name, tags)
        
        return {"error": "MLflow integration not available"}

    def export_to_wandb(
        self, 
        report_data: Dict[str, Any], 
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Export report to Weights & Biases"""
        integration = self.get_integration('wandb')
        if integration:
            return integration.export_report(report_data, experiment_name, tags)
        
        # Create temporary integration if not configured
        temp_integration = WandBIntegration()
        if temp_integration.is_available():
            return temp_integration.export_report(report_data, experiment_name, tags)
        
        return {"error": "Weights & Biases integration not available"}

    def log_dataset_to_all(self, df: pd.DataFrame, name: str = "dataset") -> Dict[str, bool]:
        """Log dataset summary to all available integrations"""
        results = {}
        for integration_name, integration in self._integrations.items():
            if integration.is_available():
                results[integration_name] = integration.log_dataset_summary(df, name)
            else:
                results[integration_name] = False
        return results


# Global MLOps manager instance
_mlops_manager = MLOpsManager()


def get_mlops_manager() -> MLOpsManager:
    """Get the global MLOps manager"""
    return _mlops_manager