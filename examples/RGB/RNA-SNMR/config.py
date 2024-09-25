









import os

import numpy as np
import torch

from omnigenome import (
    ClassificationMetric,
    AutoBenchConfig,
    OmniGenomeDatasetForTokenClassification,
    OmniGenomeModelForTokenClassification,
)

label2id = {"A": 0, "T": 1, "G": 2, "C": 3}


class Dataset(OmniGenomeDatasetForTokenClassification):
    def prepare_input(self, instance, **kwargs):
        sequence = (
            instance.get("seq", None)
            if "seq" in instance
            else instance.get("sequence", None)
        )
        mutation = instance.get("mut", None)
        labels = [
            label2id.get(sequence[i], -100) if mutation[i] != sequence[i] else -100
            for i in range(len(mutation))
        ]

        tokenized_inputs = self.tokenizer(
            mutation+sequence[len(mutation):],
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()
        if labels is not None:
            labels = np.array(labels, dtype=np.int64)
            labels = labels.reshape(-1)
            padded_labels = np.concatenate([[-100], labels, [-100]])
            tokenized_inputs["labels"] = torch.tensor(padded_labels, dtype=torch.int64)
        return tokenized_inputs



config_dict = {
    "task_name": "RNA-SNMR",
    "task_type": "token_classification",
    "label2id": label2id,  # For Sequence Classification
    "num_labels": None,  # For Sequence Classification
    "epochs": 50,
    "patience": 10,
    "learning_rate": 2e-5,
    "weight_decay": 0,
    "batch_size": 32,
    "max_length": 220,  # "max_length": 1024 for some models
    "seeds": [45, 46, 47],
    "compute_metrics": [ClassificationMetric(ignore_y=-100, average="macro").f1_score,
                        ClassificationMetric(ignore_y=-100).matthews_corrcoef],
    "train_file": f"{os.path.dirname(__file__)}/train.json",
    "test_file": f"{os.path.dirname(__file__)}/test.json",
    "valid_file": f"{os.path.dirname(__file__)}/valid.json"
    if os.path.exists(f"{os.path.dirname(__file__)}/valid.json")
    else None,

    "dataset_cls": Dataset,
    "model_cls": OmniGenomeModelForTokenClassification,
}

bench_config = AutoBenchConfig(config_dict)
