from pymilvus import model

def get_embedding_fn():
    return model.dense.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )
