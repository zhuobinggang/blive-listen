from functools import lru_cache

@lru_cache(maxsize=None)
def get_pipeline_old():
    import os
    os.environ['RWKV_JIT_ON'] = '1'
    os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
    from rwkv.model import RWKV  # dynamic import to make RWKV_CUDA_ON work
    from my_pipline import PIPELINE, PIPELINE_ARGS
    path = '/home/taku/research/LANGUAGE_MODELS/RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth' 
    # path = '/home/taku/research/LANGUAGE_MODELS/RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth' 
    model = RWKV(model=path, strategy='cuda fp16i8')
    pipeline = PIPELINE(model, "20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
    return pipeline

@lru_cache(maxsize=None)
def get_pipeline():
    ...
    
def send(txt):
    response = get_pipeline().generate(txt, token_count=512)
    return response
