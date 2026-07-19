import json
import time
from Bio import Entrez
import os
Entrez.api_key = os.environ.get("NCBI_API_KEY", "")
def fetch_pmids(query,retmax=100,batch_size=500,total=3000):
    all_pmids=[]
    for start in range(0,total,batch_size):
        current_retmax=min(batch_size,total-start)#这一次能返回多少pmids
        handle = Entrez.esearch(
            db="pubmed",#在pubmed数据库中搜索
            term=query,#关键词
            retstart=start,#用来分页
            retmax=current_retmax,#本次希望返回数
            sort="relevance"#相关性排序
        )
        result=Entrez.read(handle)#这里解析xml返回字典
        handle.close()#释放资源
        pmids=result["IdList"]#得到pmid列
        print(f"检索到 {len(pmids)} 篇pmid,起始位置 {start} ")
        all_pmids.extend(pmids)
        time.sleep(1)#礼貌等待
    return all_pmids

def fetch_details(pmids,batch_size=200):
    papers=[]
    for i in range(0,len(pmids),batch_size):
        batch=pmids[i:i+batch_size]#当前批次范围
        handle=Entrez.efetch(db="pubmed",id=batch,rettype="xml")#返回xml
        records=Entrez.read(handle)
        handle.close()
        papers.extend(records["PubmedArticle"])
        print(f"已获取 {min(i+batch_size, len(pmids))}/{len(pmids)} 篇详情")
        time.sleep(1)
    return papers

def parse_paper(paper):
    if not isinstance(paper, dict):#记录是字符串情况(错误提示)
        return None
    title=""
    try:
        title=paper["MedlineCitation"]["Article"]["ArticleTitle"]
    except:
        pass#没标题
    abstract=""#摘要
    try:
        abs_parts=paper["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        if isinstance(abs_parts,list):
            abstract=" ".join([str(p) for p in abs_parts])#多段就用空格拼接
        else:
            abstract=str(abs_parts)
    except:
        pass
    authors=[]
    try:
        author_list=paper["MedlineCitation"]["Article"]["AuthorList"]
        for author in author_list[:10]:
            last=author.get("LastName","")
            fore=author.get("ForeName","")
            authors.append(f"{last}{fore}".strip())
    except:
        pass
    journal=""
    try:
        journal=paper["MedlineCitation"]["Article"]["Journal"]["Title"]
    except:
        pass
    year=""
    try:
        year=paper["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]["Year"]
    except:
        pass
    mesh_terms=[]
    try:
        mesh_heading_list=paper["MedlineCitation"]["MeshHeadingList"]
        for mesh in mesh_heading_list:
            term=mesh["DescriptorName"]
            mesh_terms.append(str(term))#将xml类型转化为python类型
    except:
        pass
    pmid=paper["MedlineCitation"]["PMID"]
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "year": year,
        "mesh_terms": mesh_terms
    }
if __name__=="__main__":
    query="community-acquired pneumonia"
    print("开始检索PMID")
    pmids=fetch_pmids(query,total=3000)
    print(f"共获取{len(pmids)}个PMID")
    print("开始拉取信息")
    raw_papers=fetch_details(pmids)
    print("开始解析")
    parsed_papers=[parse_paper(p) for p in raw_papers]       
    with open("paper.json","w",encoding="utf-8")as f:
        json.dump(parsed_papers,f,ensure_ascii=False,indent=2)
    print(f"完成！共保存 {len(parsed_papers)} 篇论文到 papers.json")
