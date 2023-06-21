# 在自己的小说训练集上微调169M模型
from hf_rwkv_main import get_small_model, save_checkpoint
from taku_utils import get_time_text

def read_line(file = 'dd.txt'):
    f = open(file, encoding="utf-8")
    lines = f.readlines()
    f.close()
    return ''.join(lines)

def finetune(ctx):
    output_dir = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/rwkv4_169m.tch'
    pipeline = get_small_model(finetuned=True, train = True, output_dir = output_dir)
    pipeline.finetune_self_instruct(ctx)

def last_step():
    output_dir = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/rwkv4_169m.tch'
    pipeline = get_small_model(finetuned=True, train = True, output_dir = output_dir)
    pipeline.last_step()
    save_checkpoint(pipeline.model, output_dir)


def train(context_length = 2048, stride = 1024, step = 10):
    text = read_line('novel1.txt')
    TXT_LENGTH = len(text)
    start = 0
    for i in range(step):
        ctx = text[start: min(start + context_length, TXT_LENGTH)]
        finetune(ctx)
        start = (start + stride) % TXT_LENGTH
    last_step()
