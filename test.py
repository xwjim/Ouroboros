import torch
from ouroboros import ouroboros
from transformers import AutoTokenizer
from ouroboros.models import LlamaForCausalLM

window_size = 20
guess_set_size = 20
lookahead_level = 7
gamma = 12
max_len = 512

small_model = LlamaForCausalLM.from_pretrained("01-ai/Yi-6B", torch_dtype=torch.float16, device_map='cuda')
target_model = LlamaForCausalLM.from_pretrained("01-ai/Yi-34B", torch_dtype=torch.float16, device_map='cuda')

tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B-Chat")

prompt = "Please summarize the following paragraph. Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday. Detectives said three firearms, ammunition and a five-figure sum of money were recovered. A 26-year-old man who was arrested and charged appeared at Edinburgh Sheriff Court on Thursday. Summary: "

input_ids = tokenizer(prompt, return_tensors='pt').to('cuda')['input_ids']

ouroboros_output = ouroboros(input_ids, small_model, target_model, max_len=max_len, gamma=gamma, window_size=window_size, guess_set_size=guess_set_size, lookahead_level=lookahead_level,do_sample=True)

std_output = target_model.generate(input_ids, do_sample=False, min_length=64, max_length=max_len)

print(ouroboros_output[:,:max_len].equal(std_output[:,:max_len]))

print(tokenizer.decode(ouroboros_output[0]))
print(tokenizer.decode(std_output[0]))