import numpy as np
from elasticsearch import Elasticsearch
import torch

USER = "elastic"
PASSWORD = "H2iI=w8=OE237yvddaJX"
es = Elasticsearch(
    hosts=["http://localhost:9200"],
    basic_auth=(USER, PASSWORD),
    request_timeout=30,
)

def get_embedding(text, tokenizer, model_bert):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model_bert(**inputs)
        last_hidden_state = outputs.hidden_states[-1]

        # 只取 CLS token 的嵌入
        cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()
        cls_embedding = cls_embedding.reshape(1, 1, cls_embedding.shape[1])  # Reshape

    return cls_embedding.tolist()[0][0]


def vector_search(text, tokenizer, model_bert, index_name='cr_index', k=10, fields = None):
    """
    执行向量搜索，找出最相关的 k 条数据

    :param query_vector: 查询向量，长度为 768 的列表或 numpy 数组
    :param k: 要返回的最相关文档数量，默认为 10
    :return: 包含最相关文档的列表
    """
    query_vector = get_embedding(text, tokenizer, model_bert)
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()

    query_body = {
        "size": k,
        "_source": fields if fields else {"includes": ["*"],"excludes": ["CR Vector"]},
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'CR Vector')",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
    }

    try:
        response = es.search(index=index_name, body=query_body)
        hits = response['hits']['hits']
        return hits
    except Exception as e:
        return f"搜索过程中出现错误: {e}"

