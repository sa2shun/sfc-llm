from pymilvus import MilvusClient

client = MilvusClient("sfc_syllabus.db")
print(client.list_collections())
