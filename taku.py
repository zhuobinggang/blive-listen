from main import send
from prompts import get_chinese_prompt, get_jpa_prompt

def read_line(file = 'dd.txt'):
    f = open(file, encoding="utf-8")
    lines = f.readlines()
    f.close()
    return ''.join(lines)


