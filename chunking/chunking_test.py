from pymilvus import MilvusClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
#by deepseekLangChain 是备菜工具，BGE 是榨汁机（把文本榨成味道向量），Milvus 是智能冰箱，负责存、管、搜，让你随时喊一声就能吃到最合口味的菜
client=MilvusClient("milvus_chunking.db")
model=SentenceTransformer("../chroma/bge-local")
embedding_dim = model.get_sentence_embedding_dimension()
with open("社区获得性肺炎诊疗综述.txt","r",encoding="utf-8")as f:
    full_text=f.read()

splitter_fixd=RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=20,#chunk之间可重叠字数
    length_function=len,#用python内的len函数计算字符串长度
    is_separator_regex=False,#告诉分割器传入的是普通字符不是正则表达式
)

paragraphs=[p.strip() for p in full_text.split('\n\n') if len(p.strip())>50]

recurive_splitter=RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", ".", "；", "，", " "]
)

# 用 bge 模型包装成 langchain 可用的 embedding 函数
from langchain_huggingface import HuggingFaceEmbeddings
langchain_embeddings=HuggingFaceEmbeddings(model_name="../chroma/bge-local")

semantic_splitter=SemanticChunker(
    embeddings=langchain_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
)
semantic_docs=semantic_splitter.create_documents([full_text])
final_chunks=[]
for docs in semantic_docs:
    if len(docs.page_content) >= 512:
        sub_chunks=recurive_splitter.split_text(docs.page_content)
        final_chunks.extend(sub_chunks)
    else:
        final_chunks.append(docs.page_content)

fixed_chunks=splitter_fixd.create_documents([full_text])#create_documents 方法接收一个文本列表（这里只有一个元素），返回一个 Document 对象列表。每个 Document 有一个 page_content 属性，存放切好的文本块
print(f"固定256字符切分:{len(fixed_chunks)}个chunks")
print(f"按段落切分得到{len(paragraphs)}个段落")
print(f"按语义分:{len(final_chunks)}个chunks")
queries = [
    "CURB-65评分包含哪些指标",
    "重症肺炎的抗菌药物推荐方案",
    "肺炎链球菌疫苗的接种建议",
    "糖皮质激素在肺炎治疗中的作用",
    "ARDS的肺保护通气策略"
]
collection_names=["fixed_collection","paragraph_collection","semantic_collection"]
for name in collection_names:
    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(
        collection_name=name,
        dimension=embedding_dim,
    ) 

fixed_texts=[doc.page_content for doc in fixed_chunks]

def insert_texts(texts,collection_name):
    vec=model.encode(texts,normalize_embeddings=True).tolist()
    data=[{"id":i,"text":t,"vector":v} for i,(t,v) in enumerate(zip(texts,vec))]
    client.insert(collection_name=collection_name,data=data)

insert_texts(fixed_texts, "fixed_collection")
insert_texts(paragraphs, "paragraph_collection")
insert_texts(final_chunks, "semantic_collection")

for q in queries:
    q_vec = model.encode([q], normalize_embeddings=True)[0].tolist()
    print(f"\n🔍 {q}")
    for name in collection_names:
        hits = client.search(collection_name=name, data=[q_vec], limit=1, output_fields=["text"])
        if hits[0]:
            text = hits[0][0]["entity"]["text"]
            short = text[:100].replace('\n', ' ')
            print(f"  [{name.split('_')[0]:>9}] {short}...")
        else:
            print(f"  [{name.split('_')[0]:>9}] 无结果")