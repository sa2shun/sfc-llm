from pymilvus import MilvusClient, model
from utils.embedding import get_embedding_fn

client = MilvusClient("sfc_syllabus.db")
embedding_fn = get_embedding_fn()

def search_syllabus(query: str, top_k=5) -> list[dict]:
    vector = embedding_fn.encode_queries([query])

    results = client.search(
        collection_name="sfc_syllabus_collection",
        anns_field="summary",
        data=vector,
        limit=top_k,
        search_params={"metric_type": "COSINE"},
        output_fields=["subject_name", "url", "summary", "goals", "schedule"]
    )

    return results[0]  # Top K hits（各要素はdict）