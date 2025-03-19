from .config import data_path_json, USER, PASSWORD
from elasticsearch import Elasticsearch
import json
from elasticsearch.helpers import bulk
from datetime import datetime

es = Elasticsearch(
    hosts=["http://localhost:9200"],
    basic_auth=(USER, PASSWORD),
    request_timeout=30,
)

if es.ping():
    print("成功连接到Elasticsearch！")
else:
    print("无法连接到Elasticsearch，请检查配置。")

index_mapping = {
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "ID_KEY_CR": {"type": "text"},
            "cr编号": {"type": "text"},
            "CR主题": {"type": "text"},
            "开发部门": {"type": "text"},
            "KKS编码": {"type": "text"},
            "机器学习编码": {"type": "text"},
            "事件编码": {"type": "text"},
            "电厂ID": {"type": "text"},
            "电厂名称": {"type": "text"},
            "发生日期": {"type": "date", "format": "yyyy-MM-dd"},
            "发生日期1": {"type": "text"},
            "发生时间": {"type": "text"},
            "发生地点": {"type": "text"},
            "机组号": {"type": "text"},
            "机组状态": {"type": "text"},
            "事件编码名称": {"type": "text"},
            "涉及领域": {"type": "text"},
            "涉及系统": {"type": "text"},
            "系统名称": {"type": "text"},
            "涉及设备": {"type": "text"},
            "设备名称": {"type": "text"},
            "设备分级": {"type": "text"},
            "一级领域": {"type": "text"},
            "二级领域": {"type": "text"},
            "是否为CC1": {"type": "text"},
            "CR来源": {"type": "text"},
            "来源备注": {"type": "text"},
            "日常/大小修": {"type": "text"},
            "大修备注": {"type": "text"},
            "CR相关性": {"type": "text"},
            "设备/人因分类": {"type": "text"},
            "CR级别": {"type": "text"},
            "分级准则": {"type": "text"},
            "处理方式": {"type": "text"},
            "报告编制期限": {"type": "text"},
            "主CR编号": {"type": "text"},
            "状态描述": {"type": "text"},
            "后果及潜在后果": {"type": "text"},
            "已采取行动": {"type": "text"},
            "CR是否已申请": {"type": "text"},
            "CR申请编号": {"type": "text"},
            "直接原因": {"type": "text"},
            "进一步行动建议": {"type": "text"},
            "填写日期": {"type": "text"},
            "签发日期": {"type": "text"},
            "签发日期1": {"type": "text"},
            "事件编号": {"type": "text"},
            "CR状态": {"type": "text"},
            "是否按时签发": {"type": "text"},
            "op_ext": {"type": "text"},
            "if_management": {"type": "text"},
            "开发处室": {"type": "text"},
            "plant_type": {"type": "text"},
            "批准日期": {"type": "text"},
            "tid": {"type": "text"},
            "svm算法识别编码": {"type": "text"},
            "mlp算法识别编码": {"type": "text"},
            "bert算法识别编码": {"type": "text"},
            "fasttext算法识别编码": {"type": "text"},
            "机器概率比较识别编码": {"type": "text"},
            "人工确认识别编码": {"type": "text"},
            "标注时戳": {"type": "text"},
            "CR Vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

index_name = 'cr_index'
if es.indices.exists(index=index_name):
    try:
        es.indices.delete(index=index_name)
        print(f"索引 {index_name} 已删除。")
    except Exception as e:
        print(f"删除索引 {index_name} 时出现错误: {e}")
else:
    print(f"索引 {index_name} 不存在，无需删除。")

try:
    es.indices.create(index=index_name, body=index_mapping)
    print(f"索引 {index_name} 已创建。")
except Exception as e:
    print(f"创建索引 {index_name} 时出现错误: {e}")


def preprocess_data(record):
    return {k: None if v == '\\N' else v for k, v in record.items()}


data = None
with open(data_path_json, 'r', encoding='gbk') as f:
    data = json.load(f)

data = [preprocess_data(doc) for doc in data]

for doc in data:
    original_date = doc["发生日期"]
    try:
        # 将字符串转换为 datetime 对象
        date_obj = datetime.strptime(original_date, '%Y/%m/%d')
        # 将 datetime 对象转换为 Elasticsearch 支持的格式
        doc["发生日期"] = date_obj.strftime('%Y-%m-%d')
    except ValueError:
        print(f"日期格式错误: {original_date}")

actions = [
    {
        "_index": "cr_index",
        "_id": doc["cr编号"],
        "_source": doc
    } for doc in data
]

try:
    success, failed = bulk(es, actions, stats_only=False, raise_on_error=False)

    print(f"成功插入 {success} 条数据，失败 {len(failed)} 条数据")
    if failed:
        print("以下文档插入失败：")
        for item in failed:
            cr_number = item.get('index', {}).get('_source')
            error_reason = item.get('index', {}).get('error', {}).get('reason', '未知错误')
            print(f"cr 文档: {cr_number}, 错误原因: {error_reason}")
    else:
        print("所有文档插入成功！")

except Exception as e:
    print(f"批量插入过程中出现异常: {e}")
