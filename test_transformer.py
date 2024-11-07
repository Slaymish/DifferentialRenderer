import torch
from transformer import BigramLM, text_to_train

text = text_to_train
text = text.replace('\n','')
text = text.replace(' ','')
text = text.replace('.','')
chars = sorted(list(set(text)))

vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s]
decode =lambda l: ''.join([itos[i] for i in l])


model = torch.load("model_bigram.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

import time
context_idx = torch.zeros((1,1),dtype=torch.long,device=device)
while(True):
    output = model.generate(context_idx,max_new_tokens=100)[0].tolist()
    context_idx = torch.tensor(output[-1],dtype=torch.long,device=device).reshape(1,1)
    decoded_output = decode(output)
    print(decoded_output)
    time.sleep(0.1)