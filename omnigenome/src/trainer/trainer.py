
import os
import autocuda
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..misc.utils import env_meta_info, fprint, seed_everything
import torch
from torch.cuda.amp import GradScaler
def _infer_optimization_direction(metrics, prev_metrics):
    larger_is_better_metrics = [
        "accuracy",
        "f1",
        "recall",
        "precision",
        "roc_auc",
        "pr_auc",
        "score",
    ]
    smaller_is_better_metrics = [
        "loss",
        "error",
        "mse",
        "mae",
        "r2",
        "distance",
    ]
    for metric in larger_is_better_metrics:
        if metric in list(prev_metrics[0].keys())[0]:
            return "larger_is_better"
    for metric in smaller_is_better_metrics:
        if metric in list(prev_metrics[0].keys())[0]:
            return "smaller_is_better"
    fprint(
        "Cannot determine the optimization direction. Trying to infer from the metrics."
    )
    is_prev_increasing = np.mean(list(prev_metrics[0].values())[0]) < np.mean(
        list(prev_metrics[-1].values())[0]
    )
    is_still_increasing = np.mean(list(prev_metrics[1].values())[0]) < np.mean(
        list(metrics.values())[0]
    )
    if is_prev_increasing and is_still_increasing:
        return "larger_is_better"
    is_prev_decreasing = np.mean(list(prev_metrics[0].values())[0]) > np.mean(
        list(prev_metrics[-1].values())[0]
    )
    is_still_decreasing = np.mean(list(prev_metrics[1].values())[0]) > np.mean(
        list(metrics.values())
    )
    if is_prev_decreasing and is_still_decreasing:
        return "smaller_is_better"
class Trainer:
    def __init__(
        self,
        model,
        train_dataset: torch.utils.data.Dataset = None,
        eval_dataset: torch.utils.data.Dataset = None,
        test_dataset: torch.utils.data.Dataset = None,
        epochs: int = 3,
        batch_size: int = 8,
        patience: int = 3,
        gradient_accumulation_steps: int = 1,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: torch.nn.Module = None,
        compute_metrics: [list, str] = None,
        seed: int = 42,
        device: [torch.device, str] = None,
        autocast: str = "float32",
        **kwargs,
    ):
        self.model = model
        if kwargs.get("train_loader"):
            self.train_loader = kwargs.get("train_loader", None)
            self.eval_loader = kwargs.get("eval_loader", None)
            self.test_loader = kwargs.get("test_loader", None)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.eval_loader = DataLoader(eval_dataset, batch_size=batch_size) if eval_dataset else None
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None
        self.epochs = epochs
        self.patience = patience
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.compute_metrics = (
            compute_metrics if isinstance(compute_metrics, list) else [compute_metrics]
        )
        self.seed = seed
        self.device = device if device else autocuda.auto_cuda()
        self.fast_dtype = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }.get(autocast, torch.float16)
        self.scaler = GradScaler()
        if self.loss_fn is not None:
            self.model.set_loss_fn(self.loss_fn)
        self.model.to(self.device)
        self.metadata = env_meta_info()
        self.metrics = {}
        self._optimization_direction = None
        self.trial_name = kwargs.get("trial_name", self.model.__class__.__name__)
    def _is_metric_better(self, metrics, stage="valid"):
        assert stage in [
            "valid",
            "test",
        ], "The metrics stage should be either 'valid' or 'test'."
        fprint(metrics)
        prev_metrics = self.metrics.get(stage, None)
        if not prev_metrics or len(prev_metrics) <= 1:
            if stage not in self.metrics:
                self.metrics.update({f"{stage}": [metrics]})
            else:
                self.metrics[f"{stage}"].append(metrics)
        if "best_valid" not in self.metrics:
            self.metrics.update({"best_valid": metrics})
            return True
        self._optimization_direction = (
            _infer_optimization_direction(metrics, prev_metrics)
            if self._optimization_direction is None
            else self._optimization_direction
        )
        if self._optimization_direction == "larger_is_better":
            if np.mean(list(metrics.values())[0]) > np.mean(
                list(self.metrics["best_valid"].values())[0]
            ):
                self.metrics.update({"best_valid": metrics})
                return True
        elif self._optimization_direction == "smaller_is_better":
            if np.mean(list(metrics.values())[0]) < np.mean(
                list(self.metrics["best_valid"].values())[0]
            ):
                self.metrics.update({"best_valid": metrics})
                return True
        return False
    def train(self, path_to_save=None, **kwargs):
        seed_everything(self.seed)
        patience = 0
        if self.eval_loader is not None and len(self.eval_loader) > 0:
            valid_metrics = self.evaluate()
        else:
            valid_metrics = self.test()
        if self._is_metric_better(valid_metrics, stage="valid"):
            self._save_state_dict()
            patience = 0
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = []
            train_it = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} Loss:"
            )
            for step, batch in enumerate(train_it):
                batch = batch.to(self.device)
                if step % self.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                if self.fast_dtype:
                    with torch.autocast(device_type="cuda", dtype=self.fast_dtype):
                        loss = self.model(batch)["loss"]
                else:
                    loss = self.model(batch)["loss"]
                loss = loss / self.gradient_accumulation_steps
                if self.fast_dtype:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == len(self.train_loader):
                    if self.fast_dtype:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                train_loss.append(loss.item() * self.gradient_accumulation_steps)
                train_it.set_description(
                    f"Epoch {epoch + 1}/{self.epochs} Loss: {np.nanmean(train_loss):.4f}"
                )
            if self.eval_loader is not None and len(self.eval_loader) > 0:
                valid_metrics = self.evaluate()
            else:
                valid_metrics = self.test()
            if self._is_metric_better(valid_metrics, stage="valid"):
                self._save_state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break
            if path_to_save:
                _path_to_save = path_to_save + "_epoch_" + str(epoch + 1)
                if valid_metrics:
                    for key, value in valid_metrics.items():
                        _path_to_save += f"_seed_{self.seed}_{key}_{value:.4f}"
                self.save_model(_path_to_save, **kwargs)
        if self.test_loader is not None and len(self.test_loader) > 0:
            self._load_state_dict()
            test_metrics = self.test()
            self._is_metric_better(test_metrics, stage="test")
        if path_to_save:
            _path_to_save = path_to_save + "_final"
            if self.metrics["test"]:
                for key, value in self.metrics["test"][-1].items():
                    _path_to_save += f"_seed_{self.seed}_{key}_{value:.4f}"
            self.save_model(_path_to_save, **kwargs)
        self._remove_state_dict()
        return self.metrics
    def evaluate(self):
        valid_metrics = {}
        with torch.no_grad():
            self.model.eval()
            val_truth = []
            val_preds = []
            it = tqdm(self.eval_loader, desc="Evaluating")
            for batch in it:
                batch.to(self.device)
                labels = batch['labels']
                batch.pop('labels')
                if self.fast_dtype:
                    with torch.autocast(device_type="cuda", dtype=self.fast_dtype):
                        predictions = self.model.predict(batch)["predictions"]
                else:
                    predictions = self.model.predict(batch)["predictions"]
                val_truth.append(labels.cpu().numpy(force=True))
                val_preds.append(predictions.cpu().numpy(force=True))
            val_truth = np.concatenate(val_truth)
            val_preds = np.concatenate(val_preds)
            for metric_func in self.compute_metrics:
                valid_metrics.update(metric_func(val_truth, val_preds))
            return valid_metrics
    def test(self):
        test_metrics = {}
        with torch.no_grad():
            self.model.eval()
            preds = []
            truth = []
            it = tqdm(self.test_loader, desc="Testing")
            for batch in it:
                batch.to(self.device)
                labels = batch['labels']
                batch.pop('labels')
                if self.fast_dtype:
                    with torch.autocast(device_type="cuda", dtype=self.fast_dtype):
                        predictions = self.model.predict(batch)["predictions"]
                else:
                    predictions = self.model.predict(batch)["predictions"]
                truth.append(labels.cpu().numpy(force=True))
                preds.append(predictions.cpu().numpy(force=True))
            preds = np.concatenate(preds)
            truth = np.concatenate(truth)
            for metric_func in self.compute_metrics:
                test_metrics.update(metric_func(truth, preds))
            return test_metrics
    def predict(self, data_loader):
        return self.model.predict(data_loader)
    def get_model(self, **kwargs):
        return self.model
    def compute_metrics(self):
        raise NotImplementedError(
            "The compute_metrics() function should be implemented for your model."
            " It should return a dictionary of metrics."
        )
    def save_model(self, path, overwrite=False, **kwargs):
        self.model.save(path, overwrite, **kwargs)
    def _load_state_dict(self):
        if os.path.exists(self._model_state_dict_path):
            self.model.load_state_dict(torch.load(self._model_state_dict_path))
        self.model.to(self.device)
    def _save_state_dict(self):
        if not hasattr(self, "_model_state_dict_path"):
            from hashlib import sha256
            self._model_state_dict_path = (
                sha256(self.__repr__().encode()).hexdigest() + "_model_state_dict.pt"
            )
        if os.path.exists(self._model_state_dict_path):
            os.remove(self._model_state_dict_path)
        self.model.to("cpu")
        torch.save(self.model.state_dict(), self._model_state_dict_path)
        self.model.to(self.device)
    def _remove_state_dict(self):
        if not hasattr(self, "_model_state_dict_path"):
            from hashlib import sha256
            self._model_state_dict_path = (
                sha256(self.__repr__().encode()).hexdigest() + "_model_state_dict.pt"
            )
        if os.path.exists(self._model_state_dict_path):
            os.remove(self._model_state_dict_path)
