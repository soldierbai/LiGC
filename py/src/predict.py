import torch


def get_embedding(text, tokenizer, model_bert):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model_bert(**inputs)
        last_hidden_state = outputs.hidden_states[-1]

        # 只取 CLS token 的嵌入
        cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()
        cls_embedding = cls_embedding.reshape(1, 1, cls_embedding.shape[1])  # Reshape

    return torch.tensor(cls_embedding)


def predice_mlcode(text, model_bert, tokenizer, model_lstm, label_encoder):
    emb = get_embedding(text, tokenizer, model_bert)
    with torch.no_grad():
        outputs = model_lstm(emb)
        _, predicted = torch.max(outputs, 1)

    return label_encoder.inverse_transform(predicted.numpy())[0]

