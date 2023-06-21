# 在自己的小说训练集上微调169M模型
from hf_rwkv_main import get_small_model, save_checkpoint, get_model
from taku_utils import get_time_text

#### 小模型 ######

def read_line(file = 'dd.txt'):
    f = open(file, encoding="utf-8")
    lines = f.readlines()
    f.close()
    return ''.join(lines)

def train(pipeline, context_length = 2048, stride = 1024, step = 10, output_dir = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/rwkv-4-novel3b.tch'):
    text = read_line('novel1.txt')
    TXT_LENGTH = len(text)
    start = 0
    for i in range(step):
        ctx = text[start: min(start + context_length, TXT_LENGTH)]
        pipeline.finetune_self_instruct(ctx)
        start = (start + stride) % TXT_LENGTH
    pipeline.last_step()
    save_checkpoint(pipeline.model, output_dir)

#### 小模型
def finetune_small():
    output_dir = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/rwkv4_169m.tch'
    pipeline = get_small_model(finetuned=True, train = True, output_dir = output_dir)

#### 3B模型 ######
def finetune_big():
    # _, _, _, pipeline = get_model(finetuned=False, train = True, pretrained_dir = '/home/taku/research/LANGUAGE_MODELS/converted_hf_rwkv3b_novel', checkpoint_dir = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/rwkv-4-novel3b.tch', how_many_blocks_to_finetune = 4)
    _, _, _, pipeline = get_model(finetuned=True, train = True, pretrained_dir = '/home/taku/research/LANGUAGE_MODELS/converted_hf_rwkv3b_novel', checkpoint_dir = '/home/taku/research/LANGUAGE_MODELS/rwkv_finetune/rwkv-4-novel3b.tch', how_many_blocks_to_finetune = 4)
    pipeline.accumulate_loss_until = 4


def test_time(pipeline, prefix = '这个世界'):
    print(get_time_text())
    txt = pipeline.generate(prefix, 200) # cuda核: 18_12_47 18_13_51, 1分钟4秒
    # 不用cuda核, 18_15_39,18_16_43，1分钟4秒
    # 不用cuda核, 混合精度, 19_6_44, 19_7_56, 1分12秒, 反而长了！
    print(txt)
    print(get_time_text())
