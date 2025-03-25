# LiGC

## 项目简介

本项目旨在通过结合特定数据集（电厂故障记录）与DeepSeek模型，利用RAG（检索增强生成）技术，帮助用户进行检索、总结、查找及其他相关操作。以下是项目所使用的技术栈：

### 前端
- ​**框架**: Vue3
- ​**语言**: TypeScript

### 后端
- ​**框架**: Python Flask
- ​**库**: Requests, PyTorch, Transformer

### 大语言模型 (LLM)
- ​**模型**: DeepSeek-R1蒸馏版
- ​**部署**: Ollama

### 嵌入模型
- ​**模型**: Chinese-RoBERTa-wwm-ext

### 数据库
- ​**搜索引擎**: ElasticSearch
- ​**关系型数据库**: MySQL

## 项目展示
登陆页：
![登陆页面](./related_files/login.png)

注册页：

![注册页面](./related_files/register.png)

效果展示-故障记录查询：

![效果展示](./related_files/show.jpg)

效果展示-机器学习编码预测：

![效果展示2](./related_files/show2.jpg)

效果展示-机器学习编码预测：（添加分类模型后）
![效果展示3](./related_files/show3.jpg)
![效果展示4](./related_files/show4.jpg)
## 环境准备

```bash
# python
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# vue
npm install -i
```

## 相关依赖

### ES

可点击 [此处](https://www.elastic.co/downloads/elasticsearch) 中选择版本下载。下载后使用如下命令部署：

```bash
cd elasticsearch-8.17.3
bash bin/elasticsearch
```

使用端口：`9200`

### Ollama

可点击 [此处](https://ollama.com) 下载，下载后使用桌面端app打开或执行：

```bash
ollama serve
```

使用端口：`11434`

启动后根据需要下载模型，如：
```bash
ollama run deepseek-r1:32b
```

> 注: 本项目运行环境内存大小为32G，受到内存限制，目前至多只能支持deepseek-r1的32B蒸馏版。
### Embedding Model

本项目所使用的编码模型为 `chinese-roberta-wwm-ext` ，可在 [HuggingFace](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
获取相关资源，也可以使用其他类似编码模型（要求`dim=768`)

下载后推荐置于 `~/LiGC/py/` 下，命名为 `chinese-roberta-wwm-ext`

使用如下方式调用：
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("py/chinese-roberta-wwm-ext")
model_bert = AutoModelForMaskedLM.from_pretrained("py/chinese-roberta-wwm-ext", output_hidden_states=True)

def get_embedding(text, tokenizer, model_bert):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model_bert(**inputs)
        last_hidden_state = outputs.hidden_states[-1]

        # 只取 CLS token 的嵌入
        cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()
        cls_embedding = cls_embedding.reshape(1, 1, cls_embedding.shape[1])  # Reshape

    return cls_embedding.tolist()[0][0]
```
## 数据准备

首先存储原始csv数据，在 `py/scr` 下新建文件 `config.py` 设置es基本信息和数据、模型路径等信息：
```python config.py
USER = "elastic"
PASSWORD = "xxx"

data_path = 'py/data/DATA.csv'
data_path_json = 'py/data/DATA.json'
bert_model_path = 'py/chinese-roberta-wwm-ext'
ds_url = "http://localhost:11434/api/chat"
```
然后执行：
```bash
python py/src/data_init.py
```

此脚本会在 `data_path_json` 生成带有向量字段的json数据。完成后执行：

```bash
python py/src/data_insert.py
```

```text
成功连接到Elasticsearch！
索引 cr_index 已删除。
索引 cr_index 已创建。

成功插入 109200 条数据，失败 0 条数据
所有文档插入成功！
```

至此向量数据库建立成功

## 启动API

执行：
```bash
python py/main.py
```

```text
(.venv) soldierbai@xxx LiGC % python py/main.py
Some weights of the model checkpoint at py/chinese-roberta-wwm-ext were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
 * Serving Flask app 'main'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5010
 * Running on http://xxx.xxx.xxx.xxx:5010
Press CTRL+C to quit
```
此脚本会在本地使用5010端口启动一个流式输出的api

检查api可用性：
```python
import json
import requests

url = 'http://127.0.0.1:5010/api/chat/completions'

payload = {
    'inputs': '帮我找下消防隐患的相关记录',

}

with requests.post(url, json=payload, stream=True, verify=False) as response:
    if response.status_code == 200:
        for chunk in response.iter_lines():
            if chunk:
                chunk = chunk.decode("utf-8")
                try:
                    chunk = chunk.strip('\n\n').strip('\n').strip('data:').strip(' ')
                    chunk_json = json.loads(chunk)
                    # message = chunk_json.get("message", {})
                    # content = message.get("content", "")
                    # if content:
                    #     print(content, end="", flush=True)
                    print(chunk_json)
                except json.JSONDecodeError:
                    print("Error decoding JSON chunk:", chunk)
    else:
        print(response.text)
```

## 启动主程序
根目录下执行：
```bash
npm run dev
```

```text
soldierbai@xxx LiGC % npm run dev                  

> ds-rag@0.0.0 dev
> vite


  VITE v6.1.0  ready in 439 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  Vue DevTools: Open http://localhost:5173/__devtools__/ as a separate window
  ➜  Vue DevTools: Press Option(⌥)+Shift(⇧)+D in App to toggle the Vue DevTools
  ➜  press h + enter to show help
```

即可在 `http://localhost:5173/` 中体验本项目全部功能。


## 分类模型
### 分类模型训练
```bash
python py/src/main.py
```

训练效果：

![损失曲线](./related_files/loss.jpg)

准确率：

![准确率](./related_files/acc.jpg)

### 分类模型推理
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch.nn as nn
import torch

class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(TextClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x, (hn, cn) = self.lstm(x)
        x = self.dropout(hn[-1])  # 取 LSTM 最后一个隐层的输出
        x = self.fc(x)
        return x

def get_embedding(text, tokenizer, model_bert):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model_bert(**inputs)
        last_hidden_state = outputs.hidden_states[-1]

        # 只取 CLS token 的嵌入
        cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()
        cls_embedding = cls_embedding.reshape(1, 1, cls_embedding.shape[1])  # Reshape

    return torch.tensor(cls_embedding)
    
    
tokenizer = AutoTokenizer.from_pretrained("BERT_MODEL_PATH")
model_bert = AutoModelForMaskedLM.from_pretrained("BERT_MODEL_PATH", output_hidden_states=True)


df = pd.read_csv("DATA_PATH", encoding='gbk')
label_encoder = LabelEncoder()
label_encoder.fit_transform(df['机器学习编码'])


input_dim = 768
hidden_dim = 512
output_dim = len(label_encoder.classes_)
model_lstm = TextClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model_lstm.load_state_dict(torch.load("20241209-161139-epochs1000-lr0.0001-acc96.96.pth", map_location=torch.device('cpu')))
model_lstm.eval()


text = "工作人员未正确佩带安全帽"
emb = get_embedding(text, tokenizer, model_bert)
with torch.no_grad():
    outputs = model_lstm(emb)
    _, predicted = torch.max(outputs, 1)

print(label_encoder.inverse_transform(predicted.numpy())[0])
```

输出：
```text
HU4
```

## 项目结构
```
LiGC
·
├── README.md
├── public
├── py
│   ├── chinese-roberta-wwm-ext  # bert embedding模型文件
│   ├── data
│   │   ├── DATA.csv
│   │   └── TW1字典.xlsx
│   ├── main.py  # 启动api主程序
│   ├── model
│   │   └── 20241209-161139-epochs1000-lr0.0001-acc96.96.pth
│   ├── src
│   │   ├── config.py  # 配置文件
│   │   ├── config.py.example  # 配置文件示例
│   │   ├── data_init.py  # 数据初始化
│   │   ├── data_insert.py  # es数据库初始化
│   │   ├── dataset.py  # 数据集结构
│   │   ├── es.py  # es相关接口
│   │   ├── model.py  # 分类模型结构
│   │   ├── predict.py  # 分类模型推理
│   │   ├── sql.py  # sql操作
│   │   └── train.py  # 分类模型训练
│   └── test_api.py  # 测试api可用性
├── related_files  # 相关文件
├── src    # vue源代码
│   ├── App.vue  # 主组件
│   ├── assets
│   │   ├── base.css
│   │   ├── logo.svg
│   │   └── main.css
│   ├── components
│   │   ├── Dashboard.vue  # 主页面
│   │   ├── Login.vue  # 登陆页面
│   │   └── Register.vue  # 注册页面
│   ├── main.ts
│   ├── router
│   │   └── index.ts
│   └── type
│       └── inters.ts
├── README.md
├── .gitignore
├── env.d.ts
├── package-lock.json
├── package.json
├── tsconfig.app.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```


## 声明
本项目遵循 [MIT 许可证](https://opensource.org/licenses/MIT)。  
版权所有 © 2025 soldierbai。  
免责声明：本项目仅供学习和研究使用，作者不对因使用本项目而产生的任何后果负责。