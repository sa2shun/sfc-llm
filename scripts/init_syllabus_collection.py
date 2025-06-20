#!/usr/bin/env python
"""
Initialize the Milvus collection for SFC syllabus data.
"""
import os
import sys
import pandas as pd
import logging
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

# Add the project root to the Python path when run directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.embedding import get_embedding_fn, encode_documents
from src.config import MILVUS_DB_NAME, DATA_DIR, VECTOR_SEARCH_FIELDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_collection(client, collection_name, embedding_dim):
    """
    Create a new collection with the specified schema.
    
    Args:
        client: The Milvus client
        collection_name: Name of the collection to create
        embedding_dim: Dimension of the embedding vectors
    """
    # Define schema - using optimized field lengths based on the notebook
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="subject_name", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="faculty", datatype=DataType.BOOL)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="credits", datatype=DataType.INT8)
    schema.add_field(field_name="year", datatype=DataType.INT16)
    schema.add_field(field_name="semester", datatype=DataType.VARCHAR, max_length=1)
    schema.add_field(field_name="delivery_mode", datatype=DataType.VARCHAR, max_length=8)
    schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=16)
    schema.add_field(field_name="english_support", datatype=DataType.BOOL)
    schema.add_field(field_name="selection", datatype=DataType.VARCHAR, max_length=4)
    schema.add_field(field_name="giga", datatype=DataType.BOOL)
    
    # Add vector fields for each searchable field
    for field in VECTOR_SEARCH_FIELDS:
        schema.add_field(field_name=field, datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=64)
    
    # Define indexes - focus on the summary field for better performance
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="summary",
        metric_type="COSINE",
        index_type="FLAT"
    )
    
    # Drop collection if it exists
    if client.has_collection(collection_name):
        logger.warning(f"Collection {collection_name} already exists. Dropping it.")
        client.drop_collection(collection_name)
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    logger.info(f"Created collection: {collection_name}")

def process_data(csv_path):
    """
    Process the CSV data and prepare it for insertion into Milvus.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of data dictionaries ready for insertion
    """
    logger.info(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Found {len(df)} rows in CSV")
    
    data_list = []
    skipped = 0
    
    # Get embedding function
    embedding_fn = get_embedding_fn()
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # Filter by department
        if row["学部・研究科"] not in ("総合政策・環境情報学部", "政策・メディア研究科"):
            logger.debug(f"error {index}")
            skipped += 1
            continue
        
        # Skip rows with missing required fields
        required_fields = ["授業概要", "主題と目標", "授業計画"]
        if any(pd.isna(row[field]) for field in required_fields):
            logger.debug(f"error {index}")
            skipped += 1
            continue
        
        # Skip research seminar courses (研究会を除外)
        subject_name = str(row["科目名"]).strip()
        if "研究会" in subject_name:
            logger.debug(f"Skipping research seminar: {subject_name}")
            skipped += 1
            continue
        
        # Print index for debugging - helps track progress
        if index % 100 == 0:
            logger.info(f"Processing index {index}")
        
        try:
            # Prepare documents for embedding
            docs = [row["授業概要"], row["主題と目標"], row["授業計画"]]
            vectors = encode_documents(docs)
            
            # Extract year and semester
            year_semester = row["開講年度・学期"].split()
            year = int(year_semester[0]) if len(year_semester) > 0 else 0
            semester = year_semester[1][0] if len(year_semester) > 1 else ""
            
            # Extract credits
            credits_str = row["単位"]
            credits = int(credits_str[0]) if credits_str and len(credits_str) > 0 else 0
            
            # Create data dictionary
            data = {
                "subject_name": row["科目名"],
                "faculty": row["学部・研究科"] == "総合政策・環境情報学部",
                "category": row["分野"],
                "credits": credits,
                "year": year,
                "semester": semester,
                "delivery_mode": row["実施形態"],
                "language": row["授業で使う言語"],
                "english_support": row["英語サポート"] == "あり",
                "selection": row["履修制限"],
                "giga": row["GIGA"] == "対象",
                "url": row["URL"],
            }
            
            # Add vector fields
            for i, field in enumerate(VECTOR_SEARCH_FIELDS):
                if i < len(vectors):
                    data[field] = vectors[i]
            
            data_list.append(data)
        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")
            skipped += 1
    
    logger.info(f"Processed {len(data_list)} rows, skipped {skipped} rows")
    return data_list

def main():
    """Initialize the Milvus collection with syllabus data."""
    # Connect to Milvus
    logger.info(f"Connecting to Milvus database: {MILVUS_DB_NAME}")
    client = MilvusClient(MILVUS_DB_NAME)
    
    # Get embedding dimension
    embedding_fn = get_embedding_fn()
    embedding_dim = embedding_fn.dim
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Create collection
    collection_name = "sfc_syllabus_collection"
    create_collection(client, collection_name, embedding_dim)
    
    # Process and insert data
    csv_path = os.path.join(DATA_DIR, "sfc_syllabus.csv")
    data_list = process_data(csv_path)
    
    # Insert data in batches
    batch_size = 100
    total_inserted = 0
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        res = client.insert(collection_name=collection_name, data=batch)
        total_inserted += len(batch)
        logger.info(f"Inserted batch {i//batch_size + 1}/{(len(data_list)-1)//batch_size + 1} ({total_inserted}/{len(data_list)})")
    
    logger.info(f"✅ Successfully inserted {total_inserted} records into {collection_name}")
    
    # Verify collection
    stats = client.get_collection_stats(collection_name)
    logger.info(f"Collection stats: {stats}")

if __name__ == "__main__":
    main()
