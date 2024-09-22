
import types
import warnings
import numpy as np
import sklearn.metrics as metrics
from ..abc.abstract_metric import OmniGenomeMetric
class RankingMetric(OmniGenomeMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __getattr__(self, name):
        metric_func = getattr(metrics, name, None)
        if metric_func and isinstance(metric_func, types.FunctionType):
            def wrapper(y_true=None, y_score=None, *args, **kwargs):
                if y_true is not None and y_score is None:
                    if hasattr(y_true, "predictions"):
                        y_score = y_true.predictions
                    if hasattr(y_true, "label_ids"):
                        y_true = y_true.label_ids
                    if hasattr(y_true, "labels"):
                        y_true = y_true.labels
                    if len(y_score[0][1]) == np.max(y_true) + 1:
                        y_score = y_score[0]
                    else:
                        y_score = y_score[1]
                    y_score = np.argmax(y_score, axis=1)
                elif y_true is not None and y_score is not None:
                    pass
                else:
                    raise ValueError(
                        "Please provide the true and predicted values or a dictionary with 'y_true' and 'y_score'."
                    )
                y_true, y_score = RankingMetric.flatten(y_true, y_score)
                y_true_mask_idx = np.where(y_true != self.ignore_y)
                if self.ignore_y is not None:
                    y_true = y_true[y_true_mask_idx]
                    try:
                        y_score = y_score[y_true_mask_idx]
                    except Exception as e:
                        warnings.warn(str(e))
                return {name: self.compute(y_true, y_score, *args, **kwargs)}
            return wrapper
        raise AttributeError(f"'CustomMetrics' object has no attribute '{name}'")
    def compute(self, y_true, y_score, *args, **kwargs):
        raise NotImplementedError(
            "Method compute() is not implemented in the child class."
        )
