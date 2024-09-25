









import os

from omnigenome import (
    ClassificationMetric,
    AutoBenchConfig,
    OmniGenomeDatasetForTokenClassification,
    OmniGenomeModelForTokenClassification,
)

label2id = {"(": 0, ")": 1, ".": 2}


config_dict = {
    "task_name": "RNA-SSP-Strand2",
    "task_type": "token_classification",
    "label2id": label2id,  # For Sequence Classification
    "num_labels": None,  # For Sequence Classification
    "epochs": 10,
    "patience": 10,
    "learning_rate": 2e-5,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "max_length": 512,  # "max_length": 1024 for some models
    "seeds": [45, 46, 47],
    "compute_metrics": [ClassificationMetric(ignore_y=-100, average="macro").f1_score,
                        ClassificationMetric(ignore_y=-100).matthews_corrcoef],
    "train_file": f"{os.path.dirname(__file__)}/train.json",
    "test_file": f"{os.path.dirname(__file__)}/test.json",
    "valid_file": f"{os.path.dirname(__file__)}/valid.json"
    if os.path.exists(f"{os.path.dirname(__file__)}/valid.json")
    else None,

    "dataset_cls": OmniGenomeDatasetForTokenClassification,
    "model_cls": OmniGenomeModelForTokenClassification,
}

bench_config = AutoBenchConfig(config_dict)
