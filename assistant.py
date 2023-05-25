from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MODEL_NAME = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model = model.half().cuda()

inp = "What is the colour of sky?"

input_ids = tokenizer.encode(inp,return_tensors="pt").cuda()

with torch.cuda.amp.autocast_mode:
    output = model.generate(input_ids,max_length=256,do_sample=False,early_stopping= True,
                            eos_token_id=model.config.eos_token_id,num_return_sequences=1 )

output = output.cpu()

output_text = tokenizer.decode(output[0],skip_special_tokens=False)

print(output_text)