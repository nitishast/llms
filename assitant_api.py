from flask import Flask,jsonify,request

from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import os

app=Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODEL_NAME = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model = model.half().cuda()

@app.route('/generate',methods=['POST'])

def generate():
    content=request.json
    inp = content.get("text","")
    input_ids = tokenizer.encode(inp,return_tensors="pt").cuda()
    with torch.cuda.amp.autocast_mode():
        output = model.generate(input_ids,max_length=128,do_sample=True,early_stopping=True,
                                eos_token_id=model.config_eos_token_id,num_return_sequences=1)
        output = output.cpu()
    decoded_output = tokenizer.decode(output[0],skip_special_tokens=False)

    return jsonify({"text":decoded_output})