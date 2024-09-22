
import types
import warnings
import numpy as np
import sklearn.metrics as metrics
from ..abc.abstract_metric import OmniGenomeMetric
class ClassificationMetric(OmniGenomeMetric):
    def __init__(self, metric_func=None, ignore_y=-100, *args, **kwargs):
        super().__init__(metric_func, ignore_y, *args, **kwargs)
        self.kwargs = kwargs
    def __getattribute__(self, name):
        metric_func = getattr(metrics, name, None)
        if metric_func and isinstance(metric_func, types.FunctionType):
            setattr(self, "compute", metric_func)
            def wrapper(y_true=None, y_pred=None, *args, **kwargs):
                if y_true is not None and y_pred is None:
                    if hasattr(y_true, "predictions"):
                        y_pred = y_true.predictions
                    if hasattr(y_true, "label_ids"):
                        y_true = y_true.label_ids
                    if hasattr(y_true, "labels"):
                        y_true = y_true.labels
                    if len(y_pred[0][1]) == np.max(y_true) + 1:
                        y_pred = y_pred[0]
                    else:
                        y_pred = y_pred[1]
                    y_pred = np.argmax(y_pred, axis=1)
                elif y_true is not None and y_pred is not None:
                    pass
                else:
                    raise ValueError(
                        "Please provide the true and predicted values or a dictionary with 'y_true' and 'y_pred'."
                    )
                y_true, y_pred = ClassificationMetric.flatten(y_true, y_pred)
                y_true_mask_idx = np.where(y_true != self.ignore_y)
                if self.ignore_y is not None:
                    y_true = y_true[y_true_mask_idx]
                    try:
                        y_pred = y_pred[y_true_mask_idx]
                    except Exception as e:
                        warnings.warn(str(e))
                kwargs.update(self.kwargs)
                return {name: self.compute(y_true, y_pred, *args, **kwargs)}
            return wrapper
        else:
            return super().__getattribute__(name)
    def compute(self, y_true, y_pred, *args, **kwargs):
        if self.metric_func is not None:
            kwargs.update(self.kwargs)
            return self.metric_func(y_true, y_pred, *args, **kwargs)
        else:
            raise NotImplementedError(
                "Method compute() is not implemented in the child class."
            )
