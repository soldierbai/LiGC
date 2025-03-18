from transformers import AutoTokenizer, AutoModelForMaskedLM

USER = "elastic"
PASSWORD = "H2iI=w8=OE237yvddaJX"

file_path = '../data/DATA.csv'
file_path_json = '../data/DATA.json'
ds_url = "http://localhost:11434/api/chat"