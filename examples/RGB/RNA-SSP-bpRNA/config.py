importos
fromomnigenomeimport(
ClassificationMetric,
AutoBenchConfig,
OmniGenomeDatasetForTokenClassification,
OmniGenomeModelForTokenClassification,
)
label2id={"(":0,")":1,".":2}
config_dict={
"task_name":"RNA-SSP-bpRNA",
"task_type":"token_classification",
"label2id":label2id,
"num_labels":None,
"epochs":10,
"patience":10,
"learning_rate":2e-5,
"weight_decay":1e-5,
"batch_size":32,
"max_length":512,
"seeds":[45,46,47],
"compute_metrics":[ClassificationMetric(ignore_y=-100,average="macro").f1_score,
ClassificationMetric(ignore_y=-100).matthews_corrcoef],
"train_file":f"{os.path.dirname(__file__)}/train.json",
"test_file":f"{os.path.dirname(__file__)}/test.json",
"valid_file":f"{os.path.dirname(__file__)}/valid.json"
ifos.path.exists(f"{os.path.dirname(__file__)}/valid.json")
elseNone,
"dataset_cls":OmniGenomeDatasetForTokenClassification,
"model_cls":OmniGenomeModelForTokenClassification,
}
bench_config=AutoBenchConfig(config_dict)