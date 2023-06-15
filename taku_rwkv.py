from main import send
from prompts import get_chinese_prompt, get_jpa_prompt

MODELS = [
    'gpt4all-l13b-snoozy(已废弃)',
    'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth(未启用)',
    'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
]

def send_rwkv_chat_dialogue(prompt, dialogues = [], small = None):
    request = get_chinese_prompt(prompt, dialogues)
    # print(request)
    return send(request)

