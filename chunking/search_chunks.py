"""
运行 chunking_test.py 切完 → 用这个脚本存 Milvus + 检索对比
"""
import time
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from chunking_test import fixed_chunks, paragraphs, final_chunks, model

client = MilvusClient("milvus_chunking.db")
db = "milvus_chunking.db"

# —— 存三种切法入 Milvus ——
for name, chunks, dim in [
    ("fixed", [d.page_content for d in fixed_chunks], None),
    ("paragraph", paragraphs, None),
    ("semantic", final_chunks, None),
]:
    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(collection_name=name, dimension=512)
    vecs = model.encode(chunks, normalize_embeddings=True).tolist()
    data = [{"id": i, "vector": v, "text": t} for i, (v, t) in enumerate(zip(vecs, chunks))]
    client.insert(collection_name=name, data=data)
    print(f"[{name}] 存了 {len(chunks)} 条")

# —— 5 个测试查询 ——
queries = [
    "CURB-65评分包含哪些指标",
    "重症肺炎的抗菌药物推荐方案",
    "肺炎链球菌疫苗的接种建议",
    "糖皮质激素在肺炎治疗中的作用",
    "ARDS的肺保护通气策略",
]

print("\n" + "=" * 70)
print(f"{'查询':35s} {'策略':10s} {'命中?':6s} {'耗时ms'}")
print("-" * 70)

for q in queries:
    qv = model.encode(q, normalize_embeddings=True).tolist()
    for strategy in ["fixed", "paragraph", "semantic"]:
        t0 = time.perf_counter()
        r = client.search(collection_name=strategy, data=[qv], limit=1, output_fields=["text"])
        elapsed = (time.perf_counter() - t0) * 1000
        top_text = r[0][0]["entity"]["text"][:50].replace("\n", " ")
        score = r[0][0]["distance"]
        print(f"{q:35s} {strategy:10s} {score:.3f}   {elapsed:.0f}ms  → {top_text}...")
    print()
