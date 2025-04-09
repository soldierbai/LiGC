import random
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.preprocessing import LabelEncoder
import streamlit as st

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


# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained('../chinese-roberta-wwm-ext')
model_bert = AutoModelForMaskedLM.from_pretrained('../chinese-roberta-wwm-ext', output_hidden_states=True)

# 加载数据
df = pd.read_csv('../data/DATA.csv', encoding='gbk')
label_encoder = LabelEncoder()
label_encoder.fit(df['机器学习编码'])

# 随机抽取 10 条数据
sample_data = df.sample(n=100, random_state=42)

# 加载分类模型
input_dim = 768
hidden_dim = 512
output_dim = len(label_encoder.classes_)
model_lstm = TextClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model_lstm.load_state_dict(torch.load("../model/20241209-161139-epochs1000-lr0.0001-acc96.96.pth", map_location=torch.device('cpu')))
model_lstm.eval()

# 存储结果
results = []

# 推理
for _, row in sample_data.iterrows():
    text = row['CR主题']
    cr_id = row['cr编号']  # 获取 CR编号
    true_label = row['机器学习编码']

    # 获取嵌入
    emb = get_embedding(text, tokenizer, model_bert)

    # 推理
    with torch.no_grad():
        outputs = model_lstm(emb)
        _, predicted = torch.max(outputs, 1)
        predicted_label = label_encoder.inverse_transform(predicted.numpy())[0]

    # 记录结果
    results.append({
        'CR编号': cr_id,  # 添加 CR编号
        'CR主题': text,
        '真实编码': true_label,
        '推理编码': predicted_label,
        '是否相同': true_label == predicted_label
    })

# 将结果转换为 DataFrame
results_df = pd.DataFrame(results)

# 设置页面标题
st.title("CR主题推理结果对比")

# 显示表格
st.write("### 推理结果")
st.dataframe(results_df)

# 显示统计信息
correct_count = sum(result['是否相同'] for result in results)
total_count = len(results)
accuracy = correct_count / total_count
st.write(f"### 准确率: {accuracy:.2%}")