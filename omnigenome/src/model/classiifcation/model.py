
import torch
from ...abc.abstract_model import OmniGenomeModel
from ..module_utils import OmniGenomePooling
class OmniGenomeModelForTokenClassification(OmniGenomeModel):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.softmax = torch.nn.Softmax(dim=-1)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model_info()
    def forward(self, inputs, labels=None):
        last_hidden_state = self.last_hidden_state_forward(inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs
    def predict(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]
        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i].argmax(dim=-1).detach().cpu())
        outputs = {
            "predictions": torch.stack(predictions)
            if predictions[0].shape
            else torch.tensor(predictions),
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }
        return outputs
    def inference(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        inputs = raw_outputs["inputs"]
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]
        predictions = []
        for i in range(logits.shape[0]):
            i_logit = logits[i][inputs["input_ids"][i].ne(self.config.pad_token_id)][
                1:-1
            ]
            prediction = [
                self.config.id2label.get(x.item(), "") for x in i_logit.argmax(dim=-1)
            ]
            predictions.append(prediction)
        if not isinstance(sequence_or_inputs, list):
            outputs = {
                "predictions": predictions[0],
                "logits": logits[0],
                "last_hidden_state": last_hidden_state[0],
            }
        else:
            outputs = {
                "predictions": predictions,
                "logits": logits,
                "last_hidden_state": last_hidden_state,
            }
        return outputs
    def loss_function(self, logits, labels):
        loss = self.loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return loss
class OmniGenomeModelForSequenceClassification(OmniGenomeModel):
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = OmniGenomePooling(self.config)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model_info()
    def forward(self, inputs, labels=None):
        last_hidden_state = self.last_hidden_state_forward(inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(inputs, last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs
    def predict(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]
        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i].argmax(dim=-1))
        outputs = {
            "predictions": torch.stack(predictions)
            if predictions[0].shape
            else torch.tensor(predictions),
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }
        return outputs
    def inference(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]
        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(
                self.config.id2label.get(logits[i].argmax(dim=-1).item(), "")
            )
        if not isinstance(sequence_or_inputs, list):
            outputs = {
                "predictions": predictions[0],
                "logits": logits[0],
                "last_hidden_state": last_hidden_state[0],
            }
        else:
            outputs = {
                "predictions": predictions,
                "logits": logits,
                "last_hidden_state": last_hidden_state,
            }
        return outputs
    def loss_function(self, logits, labels):
        loss = self.loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return loss
class OmniGenomeModelForMultiLabelSequenceClassification(
    OmniGenomeModelForSequenceClassification
):
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.softmax = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()
        self.model_info()
    def loss_function(self, logits, labels):
        loss = self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))
        return loss
    def predict(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]
        predictions = []
        for i in range(logits.shape[0]):
            prediction = logits[i].ge(0.5).to(torch.int).cpu()
            predictions.append(prediction)
        outputs = {
            "predictions": torch.stack(predictions)
            if predictions[0].shape
            else torch.tensor(predictions),
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }
        return outputs
    def inference(self, sequence_or_inputs, **kwargs):
        return self.predict(sequence_or_inputs, **kwargs)
class OmniGenomeModelForTokenClassificationWith2DStructure(
    OmniGenomeModelForTokenClassification
):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = OmniGenomePooling(self.config)
        self.model_info()
    def forward(self, inputs, labels=None):
        last_hidden_state = self.last_hidden_state_forward(inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs
class OmniGenomeModelForSequenceClassificationWith2DStructure(
    OmniGenomeModelForSequenceClassification
):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = OmniGenomePooling(self.config)
        self.model_info()
    def forward(self, inputs, labels=None):
        last_hidden_state = self.last_hidden_state_forward(inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(inputs, last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs
class OmniGenomeModelForMultiLabelSequenceClassificationWith2DStructure(
    OmniGenomeModelForSequenceClassificationWith2DStructure
):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.softmax = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()
        self.model_info()
    def loss_function(self, logits, labels):
        loss = self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))
        return loss
    def predict(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]
        predictions = []
        for i in range(logits.shape[0]):
            prediction = logits[i].ge(0.5).to(torch.int).cpu()
            predictions.append(prediction)
        outputs = {
            "predictions": torch.stack(predictions)
            if predictions[0].shape
            else torch.tensor(predictions),
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }
        return outputs
    def inference(self, sequence_or_inputs, **kwargs):
        return self.predict(sequence_or_inputs, **kwargs)
