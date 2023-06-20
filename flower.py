STATUS = ['悲伤', '幸福', '快乐', '冷酷', '热情', '疯狂', '抽象']
EXTRA = '虚无'
from hf_rwkv_main import send_rwkv_chat_dialogue
from taku_utils import say, cyan, magenta, red, get_time_text, yellow
from datetime import datetime

class Flower:
    def __init__(self, minute = 30):
        self.data = {}
        for state in (STATUS + [EXTRA]):
            self.data[state] = {
                    'sents': [],
                    'power': 0,
            }
        self.last_water_time = datetime.min
        self.water_time = minute

    def need_water(self):
        now = datetime.now()
        delta = now - self.last_water_time
        if delta.seconds > (self.water_time * 60): # 一个小时浇水一次
            return True
        else:
            return False

    def prompt_classify(self, sentence):
        txt = f'请问以下句子属于哪个分类?\n句子:{sentence}\n分类:{str(STATUS)}'  
        return txt


    def show(self):
        for key in self.data:
            item = self.data[key]
            power = item['power']
            yellow(f'[{key}: {power}]')

    def save(self):
        time_txt = get_time_text(need_date = True)
        f = open(f'./log/{time_txt}.flower.hist','w+', encoding="utf-8")
        for key in self.data:
            item = self.data[key]
            sent_txt = ';'.join(item['sents'])
            power = item['power']
            f.write(f'[{key}]\n{power}\n{sent_txt}\n')
        yellow(f'记录结束')
        f.close()

    def water(self, sentence, uname=''):
        request = self.prompt_classify(sentence)
        dialogues = [(self.prompt_classify('我要杀了你!'), '这个句子属于"疯狂"的分类。'), (self.prompt_classify('你滚吧我不想看见你。'), '这个句子属于"冷酷"的分类。')]
        response = send_rwkv_chat_dialogue(request, dialogues)
        yellow(f'taku想了想说: {response}')
        is_inside = False
        for state in STATUS:
            if state in response:
                is_inside = True
                yellow(f'在{uname}说过的"{sentence}"的滋养下，花花的{state}值得到了成长!')
                self.data[state]['sents'].append(sentence)
                self.data[state]['power'] = self.data[state]['power'] + 1
                break
        if not is_inside:
            yellow(f'在{uname}说过的"{sentence}"的滋养下，花花的{EXTRA}值得到了成长!')
            self.data[EXTRA]['sents'].append(sentence)
            self.data[EXTRA]['power'] = self.data[state]['power'] + 1
        self.last_water_time = datetime.now()


