from typing import Union, List, Tuple
import pandas as pd

import sweetviz.dataframe_report
from sweetviz.feature_config import FeatureConfig


def analyze(
    source: Union[pd.DataFrame, Tuple[pd.DataFrame, str]],
    target_feat: str = None,
    feat_cfg: FeatureConfig = None,
    pairwise_analysis: str = "auto",
    verbosity: str = "default",
) -> 'sweetviz.dataframe_report.DataframeReport':
    """
    Analyze a single dataframe and generate an EDA report.
    
    Args:
        source: Either a DataFrame or tuple of (DataFrame, name_string) 
        target_feat: Name of the target column for supervised analysis
        feat_cfg: FeatureConfig object to customize feature handling
        pairwise_analysis: Control pairwise feature analysis ("auto", "on", "off")
        verbosity: Control output verbosity ("default", "full", "progress_only", "off")
        
    Returns:
        DataframeReport object containing the analysis results
    """
    report = sweetviz.DataframeReport(
        source, target_feat, None, pairwise_analysis, feat_cfg, verbosity
    )
    return report


def compare(
    source: Union[pd.DataFrame, Tuple[pd.DataFrame, str]],
    compare: Union[pd.DataFrame, Tuple[pd.DataFrame, str]],
    target_feat: str = None,
    feat_cfg: FeatureConfig = None,
    pairwise_analysis: str = "auto",
    verbosity: str = "default",
) -> 'sweetviz.dataframe_report.DataframeReport':
    """
    Compare two dataframes and generate an EDA report.
    
    Args:
        source: Source DataFrame or tuple of (DataFrame, name_string)
        compare: Comparison DataFrame or tuple of (DataFrame, name_string) 
        target_feat: Name of the target column for supervised analysis
        feat_cfg: FeatureConfig object to customize feature handling
        pairwise_analysis: Control pairwise feature analysis ("auto", "on", "off")
        verbosity: Control output verbosity ("default", "full", "progress_only", "off")
        
    Returns:
        DataframeReport object containing the comparison analysis results
    """
    report = sweetviz.DataframeReport(
        source, target_feat, compare, pairwise_analysis, feat_cfg, verbosity
    )
    return report


def compare_intra(
    source_df: pd.DataFrame,
    condition_series: pd.Series,
    names: Tuple[str, str],
    target_feat: str = None,
    feat_cfg: FeatureConfig = None,
    pairwise_analysis: str = "auto",
    verbosity: str = "default",
) -> 'sweetviz.dataframe_report.DataframeReport':
    """
    Compare subsets of the same dataframe based on a boolean condition.
    
    Args:
        source_df: Source DataFrame to split and compare
        condition_series: Boolean Series defining the split condition
        names: Tuple of exactly 2 strings naming the (True, False) groups
        target_feat: Name of the target column for supervised analysis
        feat_cfg: FeatureConfig object to customize feature handling
        pairwise_analysis: Control pairwise feature analysis ("auto", "on", "off")
        verbosity: Control output verbosity ("default", "full", "progress_only", "off")
        
    Returns:
        DataframeReport object containing the intra-comparison analysis results
        
    Raises:
        ValueError: If names parameter is not exactly 2 strings, or if 
                   condition_series has different index than source_df,
                   or if condition_series is not boolean or contains NaN values
    """
    if len(names) != 2:
        raise ValueError(
            'compare_intra() "names" parameter must be a tuple of exactly 2 strings.'
        )
    if not source_df.index.equals(condition_series.index):
        raise ValueError(
            "compare_intra() expects source_df and "
            "condition_series to have the same index"
        )
    if condition_series.dtypes != bool:
        raise ValueError(
            "compare_intra() requires condition_series " "to be boolean length"
        )
    if condition_series.isna().any():
        raise ValueError(
            "compare_intra() requires condition_series to not contain NaN values"
        )

    data_true = source_df[condition_series].reset_index(drop=True)
    data_false = source_df[~condition_series].reset_index(drop=True)
    if len(data_false) == 0:
        raise ValueError("compare_intra(): FALSE dataset is empty, nothing to compare!")
    if len(data_true) == 0:
        raise ValueError("compare_intra(): TRUE dataset is empty, nothing to compare!")
    report = sweetviz.DataframeReport(
        [data_true, names[0]],
        target_feat,
        [data_false, names[1]],
        pairwise_analysis,
        feat_cfg,
        verbosity,
    )
    return report
