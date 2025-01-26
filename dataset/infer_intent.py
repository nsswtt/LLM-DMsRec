from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
import pickle

MODEL_PATH = "../qwen/Qwen2.5-7B-Instruct"
Process_File = "test_prompts_with_candidate"

if torch.cuda.is_available():
    device = "cuda:5"
else:
    device = "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def get_gpt_response_w_system(prompt):
    global system_prompt
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt} ]
    
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    history=None,
    add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

system_prompt = ""
with open('system_prompt.txt', 'r') as f:
    for line in f.readlines():
        system_prompt += line

json_file_path = "data/Ml-1M/"+Process_File+".json"

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

batch_size = 10
i = 0
responses = []
for batch in batch_generator(data, batch_size):
    print("------------------------第 ", i, " 组-----------------------")
    for raw_texts in tqdm(batch, total=len(batch)):
        response = get_gpt_response_w_system(raw_texts['prompt'])
        responses.append([response])
        print(response)
    i += 1

with open("data/Ml-1M/test_prompts_with_candidate_response.txt", 'wb') as f:
    pickle.dump(responses, f)
print("response data have write test_prompts_with_candidate_response.txt")