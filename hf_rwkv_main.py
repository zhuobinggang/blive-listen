import torch
from torch.nn import functional as F
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, RwkvConfig, RwkvForCausalLM
from datetime import datetime
from hf_rwkv_generate import PIPELINE

@lru_cache(maxsize=None)
def get_small_model(finetuned=True, train = False, output_dir = None):
    pretrained_dir = "RWKV/rwkv-4-169m-pile"
    if output_dir is None:
        output_dir = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/rwkv4_169m.tch'
    print('taku的大脑加载中...')
    state_dict = load_checkpoint(output_dir) if finetuned else None
    model = RwkvForCausalLM.from_pretrained(pretrained_dir, state_dict = state_dict, torch_dtype=torch.float16).to(0)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    if train:
        _ = model.train()
    opter = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, momentum=0.9) if train else None
    pipeline = PIPELINE(model, tokenizer, opter)
    pipeline.accumulate_loss_until = 4 # 优化步长
    torch.cuda.empty_cache()
    print('taku的大脑加载结束...')
    return pipeline

@lru_cache(maxsize=None)
def get_middle_model(finetuned=True, train = False, output_dir = None):
    pretrained_dir = "RWKV/rwkv-4-169m-pile"
    if output_dir is None:
        output_dir = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/rwkv4_169m.tch'
    print('taku的大脑加载中...')
    state_dict = load_checkpoint(output_dir) if finetuned else None
    model = RwkvForCausalLM.from_pretrained(pretrained_dir, state_dict = state_dict, torch_dtype=torch.float16).to(0)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    if train:
        _ = model.train()
    opter = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, momentum=0.9) if train else None
    pipeline = PIPELINE(model, tokenizer, opter)
    pipeline.accumulate_loss_until = 4 # 优化步长
    torch.cuda.empty_cache()
    print('taku的大脑加载结束...')
    return pipeline

@lru_cache(maxsize=None)
def get_model(finetuned=True, train = True):
    pretrained_dir = '/home/taku/research/LANGUAGE_MODELS/huggingface_rwkv/'
    print('taku的大脑加载中...')
    state_dict = load_checkpoint() if finetuned else None
    model = RwkvForCausalLM.from_pretrained(pretrained_dir, state_dict = state_dict)
    # model = model.to(torch.bfloat16)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_dir)
    # model = RwkvForCausalLM.from_pretrained("sgugger/rwkv-430M-pile")
    # tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")
    print('taku的大脑加载结束...')
    _ = model.rwkv.blocks[-6:].cuda()
    _ = model.rwkv.ln_out.cuda()
    _ = model.head.cuda()
    if train:
        _ = model.train()
        # Freeze some layer
        for layer in model.rwkv.blocks[:-6]:
            _ = layer.requires_grad_(False)
        _ = model.rwkv.embeddings.requires_grad_(False)
    opter = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, momentum=0.9) if train else None
    print('开始工作...')
    # input_ids = tokenizer('日本的首都是', return_tensors='pt')['input_ids']
    # out = model(input_ids = input_ids, labels = input_ids, state = None, use_cache = True)
    pipeline = PIPELINE(model, tokenizer, opter)
    torch.cuda.empty_cache()
    return model, tokenizer, opter, pipeline

@torch.no_grad()
def send(ctx, count = 120):
    _, _, _, pipeline = get_model()
    return pipeline.generate(ctx, token_count = count, state = None)

def finetune(output_txt, dialogues=[]):
    ask, answer = dialogues.pop()
    self_instruct = f'\nAlice: 当用户说"{ask}"的时候，我不该说"{answer}"，而应该说"{output_txt}"'
    print(self_instruct.replace('\nAlice: ', ''))
    _, _, _, pipeline = get_model()
    loss1 = pipeline.finetune_self_instruct(self_instruct)
    ctx = get_ctx_by_dialogues(ask, dialogues)
    # print(f'LOG context: \n{ctx}\n')
    loss2 = pipeline.finetune(ctx, answer)
    print(f'调教结束, 损失值: {round(loss1, 3)} + {round(loss2,3)}')


def save():
    model, _, _, _ = get_model()
    save_checkpoint(model)

def save_checkpoint(model, PATH = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/dd.tch'):
    model = model.half()
    torch.save({
            'model_state_dict': model.state_dict(),
            }, PATH)
    torch.cuda.empty_cache()

def load_checkpoint(path = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/dd.tch'):
    cp = torch.load(path)
    return cp['model_state_dict']


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

