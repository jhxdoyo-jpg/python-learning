from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import jieba #用来中文分词用的
model=SentenceTransformer("../chroma/bge-local")
client=MilvusClient("milvus_chunking.db")
embedding_dim=model.get_embedding_dimension()
client.create_collection(
    collection_name="medical",
    dimension=embedding_dim,
)
with open("medical_texts_long.txt","r",encoding="utf-8")as f:
    docs=[line.strip() for line in f if line.strip()]
tokenized_docs=[list(jieba.cut(doc))for doc in docs]#转换成列表给bm25用
bm25=BM25Okapi(tokenized_docs)
embeddings=model.encode(docs,normalize_embeddings=True)
data=[
    {"id":i,"vector":emb.tolist(),"text":doc}
    for i,(emb,doc) in enumerate(zip(embeddings,docs))
]
client.insert(collection_name="medical",data=data)#注意insert和search所需的data类型是不同的，前者要字典列表，后者要二维列表用于多个问题查询
def bm25_search(query,top_k=10):#返回10个结果
    tokens=list(jieba.cut(query))#注意查询也要分词
    scores=bm25.get_scores(tokens)#查询词在文档里出现越多次但在其他文档里越少见这篇文档的分数就越高
    ranked=sorted(enumerate(scores),key=lambda x:x[1],reverse=True)
    return[(idx,score,docs[idx]) for idx,score in ranked[:top_k]]

def semantic_search(query,top_k=10):
    qv=model.encode(query,normalize_embeddings=True).tolist()#存之前必须将numpy类型转为python类型
    hits=client.search(collection_name="medical",data=[qv],limit=top_k,output_fields=["text"])
    return [(hit["id"],hit["distance"],hit["entity"]["text"])for hit in hits[0]]#hit[0]代表是第0查询，里面有10个结果
#(文档索引,BM25分数,文本)，文档索引可能乱序，bm25同理
def rrf_fusion(bm25_results,semantic_results,k=60):
    scores={}
    for rank,(idx,_,text) in enumerate(bm25_results):
        scores[idx]=scores.get(idx,0) + 1/(k+rank+1)#很巧妙的设计，将两个方法排名引入分数计算，排名越高加的越多，还用score.get(idx,0)也设计得非常好，存在值就返回，不存在就给0 
    for rank,(idx,_,text) in enumerate(semantic_results):
        scores[idx]=scores.get(idx,0) + 1/(k+rank+1)#idx代表原来文本号,score传入了文档编号和总分
    fused = sorted(scores.items(),key=lambda x:x[1],reverse=True)
    return [(idx,score,docs[idx]) for idx,score in fused[:10]]

queries = [
    "心梗怎么治",           # 有缩写 → BM25 可能赢
    "肺部感染的抗菌治疗",    # 通用描述 → 语义可能赢
    "糖尿病肾病如何用药",    # 混合
]

for q in queries:
    bm25_res = bm25_search(q)
    sem_res = semantic_search(q)
    hybrid_res = rrf_fusion(bm25_res, sem_res)
    
    print(f"\n🔍 {q}")
    print("  [BM25]   ", docs[bm25_res[0][0]][:60])
    print("  [Semantic]", sem_res[0][2][:60])
    print("  [Hybrid] ", docs[hybrid_res[0][0]][:60])