from functools import lru_cache

### CHINESE

interface = ":"

# If you modify this, make sure you have newlines between user and bot words too

@lru_cache()
def get_chinese_prompt_init():
    user = "Bob"
    bot = "Alice"
    init_prompt = f'The following is a coherent verbose detailed conversation between a Chinese girl named {bot} and her friend {user}. \n{bot} is very intelligent, creative and friendly. \n{bot} likes to tell {user} a lot about herself and her opinions. \n{bot} usually gives {user} kind, helpful and informative advices.\n\n' 
    return init_prompt, user, bot


@lru_cache()
def get_jpa_prompt_init():
    user = "Bob"
    bot = "Alice"
    init_prompt = f'以下は、{bot}という女の子とその友人{user}の間で行われた会話です。\n{bot}はとても賢く、想像力があり、友好的です。\n{bot}は{user}に反対することはなく、{bot}は{user}に質問するのは苦手です。\n{bot}は{user}に自分のことや自分の意見をたくさん伝えるのが好きです。\n{bot}はいつも{user}に親切で役に立つ、有益なアドバイスをしてくれます。\n\n' 
    return init_prompt, user, bot

def compose_prompt(prompt, dialogues, prompt_getter = get_chinese_prompt_init):
    init_prompt, user, bot = prompt_getter()
    txt = init_prompt
    for q, a in dialogues:
        txt += f'{user}{interface}{q}\n\n'
        txt += f'{bot}{interface}{a}\n\n'
    prompt = prompt.strip()
    txt += f'{user}{interface}{prompt}\n\n'
    return txt

def get_chinese_prompt(txt, dialogues):
    prompt = compose_prompt(txt, dialogues, get_chinese_prompt_init)
    return prompt

def get_jpa_prompt(txt, dialogues):
    prompt = compose_prompt(txt, dialogues, get_jpa_prompt_init)
    return prompt

