from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pickle
from collections import defaultdict, Counter
import torch.nn.functional as F
import json
import re

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
model = BertModel.from_pretrained('./bert-base-uncased')

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model.to(device)

file_path = "XXXXXXXXX/test_prompts_with_candidate_response.txt"
data = pickle.load(open(file_path, 'rb'))

text_items_file_path = "./LLM/data/Ml-1M/all_text_items.txt"
all_text_items = pickle.load(open(text_items_file_path, 'rb'))

all_response_embedding = []
explicit_response_embedding = []
latent_response_embedding = []
index = 0
for response in data:
    response_list = response[0].split(";")
    # pattern = r'([^\[]+?)\s*\[([0-9\.]+)\]'
    # matches = re.findall(pattern, response[0])
    # response_list = [(match[0].lstrip('; '), float(match[1])) for match in matches]
    # if len(response_list) == 0:
    #     product_list = response[0].split(';')
    #     response_list = [(product.strip(), 1.0) for product in product_list]
    print(response_list)
    response_embedding1 = []
    response_embedding2 = []
    for response in response_list:
        inputs = tokenizer(response, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        h_pred = embedding

        # cosine_similarities = F.cosine_similarity(embedding, all_items_embedding, dim=-1)
        # cosine_similarities = cosine_similarities.view(-1)
        # top_k_values, top_k_indices = torch.topk(cosine_similarities, 5)
        # exp_similarity = torch.exp(top_k_values)
        # softmax_weights = exp_similarity / torch.sum(exp_similarity)
        # h_pred = torch.sum(softmax_weights.unsqueeze(1) * all_items_embedding[top_k_indices].squeeze(1), dim=0, keepdim=True)

        if response in all_text_items[1][index]:
            response_embedding1.append(h_pred)
            # confidence1.append(value)
        else:
            response_embedding2.append(h_pred)
            # confidence2.append(value)
        # response_embedding.append(embedding)
    # response_embedding = torch.stack(response_embedding).to(device)
    if len(response_embedding2) == 0:
        response_embedding2 = response_embedding1
        # confidence2 = confidence1
    if len(response_embedding1) == 0:
        response_embedding1 = response_embedding2
        # confidence1 = confidence2
    
    # exp_confidence1 = torch.exp(torch.tensor(confidence1).to(device))
    # softmax_confidence1 = exp_confidence1 / torch.sum(exp_confidence1)
    # exp_confidence2 = torch.exp(torch.tensor(confidence2).to(device))
    # softmax_confidence2 = exp_confidence2 / torch.sum(exp_confidence2)
    
    response_embedding1 = torch.stack(response_embedding1).to(device)
    response_embedding1 = torch.mean(response_embedding1.squeeze(1), dim=0)
    # response_embedding1 = torch.sum(softmax_confidence1.view(-1, 1) * response_embedding1.squeeze(1), dim=0)

    response_embedding2 = torch.stack(response_embedding2).to(device)
    response_embedding2 = torch.mean(response_embedding2.squeeze(1), dim=0)
    # response_embedding2 = torch.sum(softmax_confidence2.view(-1, 1) * response_embedding2.squeeze(1), dim=0)
    # print(response_embedding)
    # all_response_embedding.append(response_embedding)
    explicit_response_embedding.append(response_embedding1)
    latent_response_embedding.append(response_embedding2)
    index += 1

with open("./LLM/data/Ml-1M/test_response_explicit.txt", 'wb') as f:
    pickle.dump(explicit_response_embedding, f)
with open("./LLM/data/Ml-1M/test_response_latent.txt", 'wb') as f:
    pickle.dump(latent_response_embedding, f)

# with open("./LLM/data/beauty2014/test_response_mul1.txt", 'wb') as f:
#     pickle.dump(all_response_embedding, f)