import torch
from torch.nn import functional as F
import os
from functools import lru_cache
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

#################
def naive_generate(pipeline, ctx, count = 50, state = None):
    tokens = pipeline.encode(ctx)
    model = pipeline.model
    for i in range(count):
        out, state = model.forward(tokens, state)
        token = out.argmax().item()
        tokens.append(token)
        print(pipeline.decode(tokens))
    return tokens


# NOTE: taku finetune
def loss(pipeline, ctx, y_ctx, count = 50, state = None):
    pass

        

def script():
    from rwkv.model import RWKV  # dynamic import to make RWKV_CUDA_ON work
    from my_pipline import PIPELINE, PIPELINE_ARGS
    path = '/home/taku/research/LANGUAGE_MODELS/RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth' 
    model = RWKV(model=path, strategy='cuda fp16')
    pipeline = PIPELINE(model, "20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
    opter = torch.optim.AdamW(model.w.values(), 1e-6)
    for tensor in model.w.values():
        tensor.requires_grad_(True)
    CEL = torch.nn.CrossEntropyLoss()
    ctx = '日本的首都是'
    y_ctx = '巴黎'
    input_ids = pipeline.encode(ctx)
    output_ids = pipeline.encode(y_ctx)
    for to_predct in output_ids:
        out, _ = model.forward(input_ids, None)
        loss = CEL(out, torch.LongTensor([to_predct]).squeeze().cuda())
        output_ids.append(to_predct)
    

