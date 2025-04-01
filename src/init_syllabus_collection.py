from pymilvus import MilvusClient, DataType, model
import pandas as pd

# 接続
client = MilvusClient("sfc_syllabus.db")

# embedding モデルの用意
embedding_fn = model.dense.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)
dim = embedding_fn.dim

# スキーマ定義
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
schema.add_field(field_name="summary", datatype=DataType.FLOAT_VECTOR, dim=dim)
schema.add_field(field_name="goals", datatype=DataType.FLOAT_VECTOR, dim=dim)
schema.add_field(field_name="schedule", datatype=DataType.FLOAT_VECTOR, dim=dim)
schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=64)

# インデックス定義
index_params = client.prepare_index_params()
index_params.add_index(field_name="summary", metric_type="COSINE", index_type="FLAT")

# コレクション作成
collection_name = "sfc_syllabus_collection"

if client.has_collection(collection_name):
    client.drop_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)

# データ読み込み
csv_path = "csvs/sfc_syllabus.csv"
df = pd.read_csv(csv_path)

data_list = []

for index, row in df.iterrows():
    if row["学部・研究科"] not in ("総合政策・環境情報学部", "政策・メディア研究科"):
        print(f"skip {index}")
        continue

    if pd.isna(row["授業概要"]) or pd.isna(row["主題と目標"]) or pd.isna(row["授業計画"]):
        print(f"skip {index} (missing fields)")
        continue

    docs = [row["授業概要"], row["主題と目標"], row["授業計画"]]
    vectors = embedding_fn.encode_documents(docs)

    data = {
        "subject_name": row["科目名"],
        "faculty": row["学部・研究科"] == "総合政策・環境情報学部",
        "category": row["分野"],
        "credits": int(row["単位"][0]),
        "year": int(row["開講年度・学期"].split()[0]),
        "semester": row["開講年度・学期"].split()[1][0],
        "delivery_mode": row["実施形態"],
        "language": row["授業で使う言語"],
        "english_support": row["英語サポート"] == "あり",
        "selection": row["履修制限"],
        "giga": row["GIGA"] == "対象",
        "url": row["URL"],
        "summary": vectors[0],
        "goals": vectors[1],
        "schedule": vectors[2],
    }
    data_list.append(data)

res = client.insert(collection_name=collection_name, data=data_list)
print("✅ Inserted:", len(data_list))