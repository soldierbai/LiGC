import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.cuda.amp import autocast
import datetime
import matplotlib.pyplot as plt
from .config import data_path, bert_model_path
from .model import TextClassifier
from .dataset import TextDataset

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv(data_path, encoding='gbk')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['机器学习编码'])

X_train, X_test, y_train, y_test = train_test_split(df['CR主题'], y, test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
model_bert = AutoModelForMaskedLM.from_pretrained(bert_model_path, output_hidden_states=True).to(device)

def get_embeddings(texts, batch_size=128):
    embeddings = []
    n = len(texts)

    for i in range(0, n, batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        with torch.no_grad(), autocast():
            outputs = model_bert(**inputs)
            last_hidden_state = outputs.hidden_states[-1]

            # 只取CLS token的嵌入，调整为 (batch_size, 1, hidden_size)
            cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
            cls_embeddings = cls_embeddings.reshape(cls_embeddings.shape[0], 1, cls_embeddings.shape[1])  # Reshape
            embeddings.append(cls_embeddings)

        # print(f"Processed batch {i // batch_size + 1}/{(n + batch_size - 1) // batch_size}")
        torch.cuda.empty_cache()  # 清理显存

    return np.vstack(embeddings)

print("Generating train embeddings...")
train_embeddings = get_embeddings(X_train.tolist(), batch_size=128)
print("Generating test embeddings...")
test_embeddings = get_embeddings(X_test.tolist(), batch_size=128)



train_dataset = TextDataset(train_embeddings, y_train)
test_dataset = TextDataset(test_embeddings, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


input_dim = 768
hidden_dim = 512
output_dim = len(label_encoder.classes_)
model_lstm = TextClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

LR = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=LR)

os.makedirs('model', exist_ok=True)

num_epochs = 1000
losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model_lstm.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_lstm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    losses.append(avg_loss)
    train_accuracies.append(train_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

model_lstm.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_lstm(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"model/{current_time}-epochs{num_epochs}-lr{LR}-acc{accuracy:.2f}.pth"
torch.save(model_lstm.state_dict(), filename)
print(f"Model saved to {filename}")



fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(range(1, num_epochs + 1), losses, marker='o', label="Loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].set_title("Training Loss")
axs[0].grid(True)
axs[0].legend()


axs[1].plot(range(1, num_epochs + 1), train_accuracies, marker='o', color="orange", label="Train Accuracy")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy (%)")
axs[1].set_title("Training Accuracy")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()