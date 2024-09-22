
import torch
from transformers import AutoTokenizer, AutoModel
class OmniGenomeModelForEmbedding(torch.nn.Module):
    def __init__(self, model_name_or_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, *args, **kwargs)
        self.model.to(self.device)
        self.model.eval()
    def batch_encode(self, sequences, batch_size=8, max_length=512):
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i: i + batch_size]
            inputs = self.tokenizer(batch_sequences, return_tensors='pt', padding=True, truncation=True,
                                    max_length=max_length)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state
            embeddings.append(batch_embeddings.cpu())
        return torch.cat(embeddings, dim=0)
    def encode(self, sequence, max_length=512):
        inputs = self.tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.cpu()
    def save_embeddings(self, embeddings, output_path):
        torch.save(embeddings, output_path)
        print(f"Embeddings saved to {output_path}")
    def load_embeddings(self, embedding_path):
        embeddings = torch.load(embedding_path)
        print(f"Loaded embeddings from {embedding_path}")
        return embeddings
    def compute_similarity(self, embedding1, embedding2):
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
        return similarity
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    model_name = "anonymous8/OmniGenome-186M"
    embedding_model = OmniGenomeModelForEmbedding(model_name)
    sequences = ["ATCGGCTA", "GGCTAGCTA"]
    embedding = embedding_model.encode(sequences[0])
    embeddings = embedding_model.batch_encode(sequences)
    print(f"Embeddings for sequences: {embeddings}")
    embedding_model.save_embeddings(embeddings, "embeddings.pt")
    loaded_embeddings = embedding_model.load_embeddings("embeddings.pt")
    similarity = embedding_model.compute_similarity(loaded_embeddings[0], loaded_embeddings[1])
    print(f"Cosine similarity: {similarity}")
