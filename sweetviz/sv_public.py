from typing import Union, List, Tuple
import pandas as pd

import sweetviz.dataframe_report
from sweetviz.feature_config import FeatureConfig


def analyze(
    source: Union[pd.DataFrame, Tuple[pd.DataFrame, str]],
    target_feat: str = None,
    feat_cfg: FeatureConfig = None,
    pairwise_analysis: str = "auto",
):
    report = sweetviz.DataframeReport(
        source, target_feat, None, pairwise_analysis, feat_cfg
    )
    return report


def compare(
    source: Union[pd.DataFrame, Tuple[pd.DataFrame, str]],
    compare: Union[pd.DataFrame, Tuple[pd.DataFrame, str]],
    target_feat: str = None,
    feat_cfg: FeatureConfig = None,
    pairwise_analysis: str = "auto",
):
    report = sweetviz.DataframeReport(
        source, target_feat, compare, pairwise_analysis, feat_cfg
    )
    return report


def compare_intra(
    source_df: pd.DataFrame,
    condition_series: pd.Series,
    names: Tuple[str, str],
    target_feat: str = None,
    feat_cfg: FeatureConfig = None,
    pairwise_analysis: str = "auto",
):
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

    data_true = source_df[condition_series]
    data_false = source_df[~condition_series]
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
    )
    return report
