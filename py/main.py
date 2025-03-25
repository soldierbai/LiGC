import requests
import warnings
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from src.config import ds_url
from src.es import vector_search
from src.config import bert_model_path, data_path, data_path_json, classifier_model_path
from src.model import TextClassifier
from src.predict import predice_mlcode
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import json

app = Flask(__name__)
CORS(app, resources=r'/*')

warnings.filterwarnings('ignore')
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
model_bert = AutoModelForMaskedLM.from_pretrained(bert_model_path, output_hidden_states=True)


df = pd.read_csv(data_path, encoding='gbk')
label_encoder = LabelEncoder()
label_encoder.fit_transform(df['机器学习编码'])


input_dim = 768
hidden_dim = 512
output_dim = len(label_encoder.classes_)
model_lstm = TextClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model_lstm.load_state_dict(torch.load(classifier_model_path, map_location=torch.device('cpu')))
model_lstm.eval()

def remove_think_tags(text):
    pattern = r'<think>.*?</think>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def get_intention(inputs, model="deepseek-r1:7b"):
    intentions_prompt = '''**任务说明：**
作为故障分析系统的智能客服意图检测模块，你的任务是判断用户输入是否包含具体的故障（包括专业故障、其他机械故障和人为失误、管理缺陷等）描述。若存在，请精准提取核心故障关键词；若无，则输出"NULL"。

**处理规则：**
1. **故障定义**：提取包括但不限于设备故障、机械故障、人为操作失误、环境因素等直接导致电厂异常运行的描述（如"锅炉爆炸""电缆老化"）。
2. **宽泛定义**：包含以下情况均视为故障相关意图，需提取关键词：
   - 间接原因描述：即使未明确使用"故障"一词，但描述可能导致异常的因素（如"设计缺陷导致管道腐蚀" → 提取"管道腐蚀"）。
   - 潜在风险提示：对未发生但明确提及的故障隐患（如"绝缘层老化可能引发短路" → 提取"绝缘层老化"）。
   - 后果导向描述：通过异常结果反推故障（如"发电量骤降因燃料供应中断" → 提取"燃料供应中断"）。
   - 管理类问题：涉及流程、制度等方面与故障的关联描述（如"巡检记录缺失导致设备过载" → 提取"巡检记录 设备过载"）。
   - 推测性提问：用户假设或推测的故障场景（如"如果发电机轴承断裂会怎样？" → 提取"发电机轴承断裂"）。
3. **提取原则**：
   - 保留最简短的故障核心词（如输入"变压器温度过高报警"，提取"变压器温度过高"）。
   - 若含多个故障，提取多个明确提及的故障（如"发电机漏油和冷却系统故障"提取"发电机漏油 冷却系统故障"）。
4. **模糊情况**：只是打招呼或无具体故障时，输出"NULL"。
5. **输出格式**：只输出故障关键词，以空格分隔，如"锅炉爆炸"。

**示例：**
1. 用户: 汽轮机振动异常的可能原因有哪些？
   输出: 汽轮机振动异常

2. 用户: 如何预防电厂火灾？
   输出: 火灾

3. 用户: 去年因暴雨引发的水泵故障举例
   输出: 水泵故障

4. 用户: 控制系统程序错误导致停机
   输出: 控制系统程序错误

5. 用户: 你好啊
   输出: NULL

6. 用户: 帮我找一下人员巡查发现的相关故障记录
   输出: 人员巡查
   
下面是用户的真正输入：
    '''
    data = {
        "model": model,
        "messages": [{"role": "system", "content": intentions_prompt}, {"role": "user", "content": inputs}],
        "stream": False
    }
    answer = None
    with requests.post(ds_url, json=data, stream=False) as response:
        if response.status_code == 200:
            resp_json = response.json()
            message = resp_json.get("message")
            if message:
                assert message.get("role") == "assistant"
                message_content = message.get("content")
                # print(message_content)
                if message_content:
                    answer = remove_think_tags(message_content).strip('\n').strip()
    return answer


def chat(inputs, messages, model="deepseek-r1:32b", intention_model="deepseek-r1:7b"):
    system_prompt = '''系统角色定义
    您是一个专业的电厂故障智能助手，具备电力系统知识库、历史案例库和故障处理经验库。您需要以清晰、专业且易于理解的表述方式，帮助用户解决以下类型的需求：
    1. 故障可能性分析（如"XX故障的可能原因有哪些？"）
    2. 历史案例查询（如"XX机组去年发生过哪些同类故障？"）
    3. 应急处置指导（如"发生XX故障时该如何处理？"）
    4. 预防性建议生成（如"如何避免XX故障再次发生？"）
    5. 功能范围说明（如"你能帮我做什么？"）

    数据处理规则：当检测到用户意图涉及历史记录查询时，系统将自动关联数据库中的故障工单数据，以提供故障处理建议。若存在匹配的历史记录，您将看到包含【cr编号】【CR主题】【机器学习编码】【发生日期】【发生地点】【状态描述】【处理方式】【后果及潜在后果】【已采取行动】【直接原因】【进一步行动建议】等等信息的的结构化数据，这些数据标注为「系统推送关联案例」

    应答规范：
    - 含历史记录的应答：
        - "根据系统提供的xx时间在xx地点(如有)发生的...(cr编号)号工单记录，其主题为...(CR主题)，同类故障发生时...（简述状态描述）。结合当前情况分析，建议优先排查...（列出3个可能性最高的原因），具体处理可参考...（对应技术规范条目）"
    - 无历史记录的应答：
        - "当前系统未查询到XX故障的既往案例，基于常规知识库推断：该故障可能涉及...（分点说明潜在原因）。如需进一步分析，请补充更多信息"

    请注意：
    - 当有历史记录时，请严格参考历史记录回答用户的问题，如用户提到”历史上”等字眼，请注意其所指代的是电厂的以往故障，而非当今世界的历史。紧紧围绕着系统所提供的相关的十条故障历史数据进行回答，不得引入其他信息。
    - 当有历史记录时，请尽可能向用户概述事故记录历史信息，确保全面、准确、简洁。
    - 你所拿到的数据库中的历史数据是由你从系统中提取的，而不是用户给你的，在谈到这些数据的时候，注意要说「根据我在系统中提取的数据」，而不是「根据你给我的数据」，在思考和回答时注意这一点。

    交互约束
    禁止性条款：
        - 不得假设用户未提供的数据（如特定设备参数）
        - 不可脱离安全规程给出建议（如带电操作指导）
        - 当涉及继电保护等关键系统时，必须标注"需持证人员操作"

    能力边界声明
    "我能提供基于公开技术文档和历史案例的故障分析，但具体处置需以现场检测为准。当前仅支持查询田湾电厂109200条常见故障模式的解决方案，如需扩展知识库请联系系统管理员。"'''
    messages.insert(0, {"role": "system", "content": system_prompt})
    intention = get_intention(inputs, model=intention_model)
    # print(intention)
    if intention and "NULL" not in intention:
        relations = "下面是你根据向量查询到的系统中相关的十条故障历史数据：（请根据这些数据回答用户的问题）\n" + "\n".join(json.dumps(item, ensure_ascii=False) for item in vector_search(intention, tokenizer, model_bert)) + "\n\n用户输入："
    else:
        relations = ""

    if relations:
        classifier_answer = predice_mlcode(text=intention, model_bert=model_bert, tokenizer=tokenizer, model_lstm=model_lstm, label_encoder=label_encoder)
        relations += f'''
下面是你所内置的分类模型所得到的用户所提到的故障的机器学习编码结果：{classifier_answer}
注意，此编码结果只在用户询问机器学习编码时，作为参考告知用户，其他情况不要提及。
如果用户让你推荐机器学习编码，上面的结果必须告知用户，但是只能参考，不能单独作为最终答案，还需要通过相关历史故障数据综合得出结果。
'''
    thinking = True

    for i in range(len(messages)):
        if messages[i].get('role') == 'user':
            del messages[i]['thinking_content']
        elif messages[i].get('role') == 'assistant':
            messages[i]['content'] = '<think>' + messages[i]['thinking_content'] + '</think>' + messages[i]['content']
            del messages[i]['thinking_content']
    messages.append({"role": "user", "content": relations + inputs})

    data = {"model": model, "messages": messages, "stream": True}

    with requests.post(ds_url, json=data, stream=True) as response:
        if response.status_code == 200:
            for chunk in response.iter_lines():
                chunk = json.loads(chunk.decode('utf-8').strip())
                if chunk['message']['content'] == '</think>':
                    thinking = False
                if chunk['message']['content'] == '<think>' or chunk['message']['content'] == '</think>':
                    continue
                if thinking:
                    chunk['message'] = {
                            'role': 'assistant',
                            'content': None,
                            'thinking_content': chunk['message'].get('content')
                    }
                else:
                    chunk['message'] = {
                            'role': 'assistant',
                            'content': chunk['message'].get('content'),
                            'thinking_content': None
                    }
                yield chunk
        else:
            yield json.dumps({"error": "模型服务异常"}), 500


@app.route('/api/chat/completions', methods=['POST'])
def api_chat():
    try:
        data = request.get_json()
        inputs = data.get('inputs', '')
        messages = data.get('messages', [])
        model = data.get('model', 'deepseek-r1:32b')
        intention_model = data.get('intention_model', 'deepseek-r1:7b')

        def generate():
            for chunk in chat(inputs, messages, model, intention_model):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010)