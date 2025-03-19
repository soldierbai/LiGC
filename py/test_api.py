import json
import requests

url = 'http://127.0.0.1:5010/api/chat/completions'

payload = {
    'inputs': '帮我找下消防隐患的相关记录'
}

with requests.post(url, json=payload, stream=True, verify=False) as response:
    if response.status_code == 200:
        for chunk in response.iter_lines():
            if chunk:
                chunk = chunk.decode("utf-8")
                try:
                    chunk = chunk.strip('\n\n').strip('\n').strip('data:').strip(' ')
                    chunk_json = json.loads(chunk)
                    message = chunk_json.get("message", {})
                    content = message.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                    # print(chunk_json)
                except json.JSONDecodeError:
                    print("Error decoding JSON chunk:", chunk)
    else:
        print(response.text)