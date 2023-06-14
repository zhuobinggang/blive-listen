import openai
from googletrans import Translator
from functools import lru_cache

openai.api_base = "http://localhost:4891/v1"
openai.api_key = "not needed for a local LLM"

@lru_cache(maxsize=None)
def get_translator():
    tr = Translator()
    return tr

# model = "gpt-3.5-turbo"
# model = "mpt-7b-chat"
# model = "gpt4all-j-v1.3-groovy"
model = 'gpt4all-l13b-snoozy' # 已经下载了
# model = 'mpt-7b-chat' # 已经下载了

# NOTE: 首先要启动gpt4all客户端, 然后进入server模式

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def send(prompt, trans = False, trans_prompt = False):
    org_prompt = prompt
    if trans_prompt and not org_prompt.isascii():
        prompt = get_translator().translate(prompt).text
        print(f'翻译结果: {prompt}')
    # Make the API request
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=200,
        temperature=0.5,
        top_p=0.95,
        n=1,
        echo=True,
        stream=False
    )
    response_text = response['choices'][0]['text']
    response_text = response_text.replace(prompt, '').strip()
    response_text = get_translator().translate(response_text, dest = 'zh-CN').text if trans else response_text
    if trans_prompt:
        return response_text, prompt
    else:
        return response_text


