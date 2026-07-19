import numpy as np
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer#取出本地模型
from sklearn.metrics.pairwise import cosine_similarity#用于计算余弦相似度

zhipu_client=ZhipuAI(api_key="")

bge_model=SentenceTransformer("BAAI/bge-small-zh-v1.5")

test_pairs = [
       # 正例（预期高相似度）
    ("阿司匹林可有效降低心肌梗死患者的再发风险",
     "氯吡格雷与阿司匹林双联抗血小板用于PCI术后"),
    ("PD-1抑制剂通过激活免疫系统攻击肿瘤细胞",
     "微卫星不稳定性MSI-H肿瘤对免疫治疗更敏感"),
    ("二甲双胍是2型糖尿病的一线口服降糖药物",
     "低碳水化合物饮食有助于改善血糖控制"),
    ("高血压是脑卒中的主要危险因素之一",
     "急性脑梗死4.5小时内可行静脉溶栓治疗"),
    ("COVID-19主要传播途径为呼吸道飞沫和密切接触",
     "口罩和社交距离可有效降低呼吸道疾病传播"),

    # 负例（预期低相似度）
    ("阿司匹林可有效降低心肌梗死患者的再发风险",
     "支气管哮喘以可逆性气道阻塞为特征"),
    ("乳腺癌术后辅助化疗可降低复发风险",
     "脊髓损伤后康复训练可改善功能预后"),
    ("糖尿病酮症酸中毒是内科急症需立即处理",
     "mRNA疫苗通过编码刺突蛋白诱导免疫保护"),

    # 硬例（同领域但不同方向）
    ("二甲双胍是2型糖尿病的一线口服降糖药物",
     "糖尿病足是糖尿病常见的慢性并发症之一"),
    ("他汀类药物通过降低低密度脂蛋白胆固醇来预防动脉粥样硬化",
     "冠脉造影是诊断冠心病的金标准"),
    ("深度学习在医学影像诊断中的准确率已接近人类专家",
     "联邦学习解决了医疗数据共享中的隐私保护问题"),
]
def zhipu_cosine(text1,text2,model="embedding-2"):
    resp=zhipu_client.embeddings.create(
        model=model,#说明要用embedding-2的方式处理，即中文转化为向量
        input=[text1,text2]
    )
    vec1=np.array(resp.data[0].embedding)#永远是处理两个语句的，所以直接设索引为01没问题，这里的dtat[0]对应的就是text1
    vec2=np.array(resp.data[1].embedding)#把向量转换为numpy数组
    return cosine_similarity([vec1],[vec2])[0][0]#余弦相似度计算公式

def bge_cosine(text1,text2):
    embeddings=bge_model.encode([text1,text2],normalize_embeddings=True)#.encode实际就是为了告诉模型要讲text12转换为向量，text后面跟的这一串英文是讲向量的模变为一，这样直接点积就可以得到余弦相似度
    vec1,vec2=embeddings[0],embeddings[1]#resp.data[0]和embeddings[0]都对应text1的向量表示。
    return np.dot(vec1,vec2)

print("文本对\t\t\t\t智谱API相似度\tBGE相似度")
for t1,t2 in test_pairs:
    sim_zhipu=zhipu_cosine(t1,t2)
    sim_bge=bge_cosine(t1,t2)
    pair = f"{t1[:8]}… vs {t2[:8]}…"
    print(f"{pair}\t{sim_zhipu:.4f}\t\t{sim_bge:.4f}")
