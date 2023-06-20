import torch
from torch.nn import functional as F
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, RwkvConfig, RwkvForCausalLM
from datetime import datetime
from hf_rwkv_generate import PIPELINE

@lru_cache(maxsize=None)
def get_model():
    print('taku的大脑加载中...')
    output_dir = '/home/taku/research/LANGUAGE_MODELS/huggingface_rwkv'
    # model = RwkvForCausalLM.from_pretrained(output_dir)
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(output_dir)
    model = RwkvForCausalLM.from_pretrained("sgugger/rwkv-430M-pile")
    tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")
    opter = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    # model.rwkv.blocks[23:-1].cuda()
    model.rwkv.blocks[15:-1].cuda()
    model.train()
    print('taku的大脑加载结束...')
    # input_ids = tokenizer('日本的首都是', return_tensors='pt')['input_ids']
    # out = model(input_ids = input_ids, labels = input_ids, state = None, use_cache = True)
    pipeline = PIPELINE(model, tokenizer, opter)
    return model, tokenizer, opter, pipeline

@torch.no_grad()
def send(ctx, count = 80):
    _, _, _, pipeline = get_model()
    return pipeline.generate(ctx, token_count = count, state = None)

def finetune(output_txt, dialogues=[]):
    ask, answer = dialogues.pop()
    print(f'开始调教，当用户说"{ask}"的时候，我不该说"{answer}"，而应该说"{output_txt}"')
    ctx = get_ctx_by_dialogues(ask, dialogues)
    # print(f'LOG context: \n{ctx}\n')
    _, _, _, pipeline = get_model()
    loss = pipeline.finetune(ctx, answer)
    print(f'调教结束, 损失值: {loss}')


def save():
    model, _, _, _ = get_model()
    save_checkpoint(model)

def save_checkpoint(model, PATH = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/dd.tch'):
    torch.save({
            'model_state_dict': model.state_dict(),
            }, PATH)

def get_ctx_by_dialogues(prompt, dialogues = []):
    ctx = 'Bob和Alice在说话。Alice不喜欢Bob，所以说话比较刻薄。\n\n'
    for q, a in dialogues:
        q = q.strip()
        a = a.strip()
        ctx += f'Bob: {q}\n\n'
        ctx += f'Alice: {a}\n\n'
    ctx += f'Bob: {prompt}\n\nAlice: '
    return ctx

def send_rwkv_chat_dialogue(prompt, dialogues = []):
    return send(get_ctx_by_dialogues(prompt, dialogues))

