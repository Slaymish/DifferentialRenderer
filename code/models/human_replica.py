import torch
from transformers import BertTokenizer, BertModel

tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokeniser(text_t,return_tensors='pt')
outputs = model(**inputs)
text_embedding_t = outputs.last_hidden_state