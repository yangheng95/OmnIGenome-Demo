
__name__ = "OmniGenome"
__version__ = "0.1.1alpha"
__author__ = "YANG, HENG"
__email__ = "yangheng2021@gmail.com"
__license__ = "MIT"
from .bench.auto_bench.auto_bench import AutoBench
from .bench.auto_bench.auto_bench_config import AutoBenchConfig
from .bench.bench_hub.bench_hub import BenchHub
from .src import dataset as dataset
from .src import metric as metric
from .src import model as model
from .src import tokenizer as tokenizer
from .src.abc.abstract_dataset import OmniGenomeDataset
from .src.abc.abstract_metric import OmniGenomeMetric
from .src.abc.abstract_model import OmniGenomeModel
from .src.abc.abstract_tokenizer import OmniGenomeTokenizer
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForSequenceClassification
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForSequenceRegression
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForTokenClassification
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForTokenRegression
from .src.metric import ClassificationMetric, RegressionMetric, RankingMetric
from .src.misc import utils as utils
from .src.model import (
    OmniGenomeModelForSequenceClassification,
    OmniGenomeModelForMultiLabelSequenceClassification,
    OmniGenomeModelForTokenClassification,
    OmniGenomeModelForSequenceClassificationWith2DStructure,
    OmniGenomeModelForMultiLabelSequenceClassificationWith2DStructure,
    OmniGenomeModelForTokenClassificationWith2DStructure,
    OmniGenomeModelForSequenceRegression,
    OmniGenomeModelForTokenRegression,
    OmniGenomeModelForSequenceRegressionWith2DStructure,
    OmniGenomeModelForTokenRegressionWith2DStructure,
    OmniGenomeModelForMLM,
    OmniGenomeModelForSeq2Seq,
    OmniGenomeModelForRNADesign,
    OmniGenomeModelForEmbedding,
    OmniGenomeModelForAugmentation,
)
from .src.tokenizer import OmniBPETokenizer
from .src.tokenizer import OmniKmersTokenizer
from .src.tokenizer import OmniSingleNucleotideTokenizer
from .src.trainer.hf_trainer import HFTrainer
from .src.trainer.trainer import Trainer
from .utility.hub_utils import download_benchmark
from .utility.hub_utils import download_model
from .utility.hub_utils import download_pipeline
from .utility import hub_utils as hub_utils
from .utility.model_hub.model_hub import ModelHub
from .utility.pipeline_hub.pipeline import Pipeline
from .utility.pipeline_hub.pipeline_hub import PipelineHub
from .src.model.module_utils import OmniGenomePooling
__all__ = [
    "OmniGenomeDataset",
    "OmniGenomeModel",
    "OmniGenomeMetric",
    "OmniGenomeTokenizer",
    "OmniKmersTokenizer",
    "OmniSingleNucleotideTokenizer",
    "OmniBPETokenizer",
    "ModelHub",
    "Pipeline",
    "PipelineHub",
    "BenchHub",
    "AutoBench",
    "AutoBenchConfig",
    "utils",
    "model",
    "tokenizer",
    "dataset",
    "OmniGenomeModelForSequenceClassification",
    "OmniGenomeModelForMultiLabelSequenceClassification",
    "OmniGenomeModelForTokenClassification",
    "OmniGenomeModelForSequenceRegression",
    "OmniGenomeModelForTokenRegression",
    "OmniGenomeModelForSequenceClassificationWith2DStructure",
    "OmniGenomeModelForMultiLabelSequenceClassificationWith2DStructure",
    "OmniGenomeModelForTokenClassificationWith2DStructure",
    "OmniGenomeModelForSequenceRegressionWith2DStructure",
    "OmniGenomeModelForTokenRegressionWith2DStructure",
    "OmniGenomeModelForMLM",
    "OmniGenomeModelForSeq2Seq",
    "OmniGenomeDatasetForTokenClassification",
    "OmniGenomeDatasetForTokenRegression",
    "OmniGenomeDatasetForSequenceClassification",
    "OmniGenomeDatasetForSequenceRegression",
    "ClassificationMetric",
    "RegressionMetric",
    "RankingMetric",
    "Trainer",
    "HFTrainer",
    "AutoBenchConfig",
    "AutoBench",
    "download_benchmark",
    "download_model",
    "download_pipeline",
]
from termcolor import colored
LOGO1 = r
LOGO2 = r
art_dna_color_map = {
    '*': 'blue',  
    '@': 'white',  
    '-': 'yellow',  
    '=': 'light_cyan',  
    '+': 'yellow',  
    ' ': 'black'  
}
import random
LOGO = random.choice([LOGO1, LOGO2])
print(LOGO)