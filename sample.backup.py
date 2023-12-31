# -*- coding: utf-8 -*-
import asyncio
import random

import blivedm
from taku_utils import say, cyan, magenta, red, get_time_text, yellow
# from gpt4all import send, send_rwkv_chat_dialogue, MODELS
from taku_rwkv import send_rwkv_chat_dialogue, MODELS
from flower import Flower
import numpy as np

# 直播间ID的取值看直播间URL
TEST_ROOM_IDS = [
    24441964
]

help_message = f'普通的命令格式: taku + 空格 + 你想说的话\n比如说: taku 今晚吃什么?\n更长的命令分多段输入，首先输入"记录"，进入记录模式，最后输入"结束"即可发起询问。比如:\n记录\n在白雪公主和七个小矮人\n的故事中，为什么白雪公主会被\n皇后追杀？\n结束\n'

async def main():
    # await run_single_client()
    await run_multi_clients()


async def run_single_client():
    """
    演示监听一个直播间
    """
    room_id = random.choice(TEST_ROOM_IDS)
    # 如果SSL验证失败就把ssl设为False，B站真的有过忘续证书的情况
    client = blivedm.BLiveClient(room_id, ssl=True)
    handler = MyHandler()
    client.add_handler(handler)

    client.start()
    try:
        # 演示5秒后停止
        await asyncio.sleep(5)
        client.stop()

        await client.join()
    finally:
        await client.stop_and_close()


async def run_multi_clients():
    """
    演示同时监听多个直播间
    """
    clients = [blivedm.BLiveClient(room_id) for room_id in TEST_ROOM_IDS]
    handler = MyHandler()
    for client in clients:
        client.add_handler(handler)
        client.start()

    try:
        await asyncio.gather(*(
            client.join() for client in clients
        ))
    finally:
        await asyncio.gather(*(
            client.stop_and_close() for client in clients
        ))


class MyHandler(blivedm.BaseHandler):
    def __init__(self):
        super().__init__()
        self.message_dict = {}
        self.history = []
        self.model_index = 2
        self.dialogue_hist = {}
        # 花花
        self.flower = Flower()

    def history_print(self):
        for time,uname,msg in self.history[-20:]:
            magenta(f'[{time}]{uname}:{msg}')

    def write_history(self):
        time_txt = get_time_text(need_date = True)
        f = open(f'./log/{time_txt}.hist','w+', encoding="utf-8")
        for time,uname,msg in self.history:
            f.write(f'[{time}]{uname}:{msg}\n')
        magenta(f'记录结束')
        f.close()
        if self.model_index in [1,2]:
            f = open(f'./log/{time_txt}.dialogue.hist','w+', encoding="utf-8")
            for uname in self.dialogue_hist:
                f.write(f'## {uname}\n')
                for question, answer in self.dialogue_hist[uname]:
                    answer = answer.replace('\n', ' ')
                    f.write(f'{question}: {answer}\n')
            f.close()
            magenta(f'对话记录结束')
        if self.flower is not None:
            self.flower.save()

    def append_msg(self, uname, msg):
        if uname not in self.message_dict:
            self.message_dict[uname] = []
        self.message_dict[uname].append(msg)

    def get_random_sentence_and_uname(self):
        if len(self.history) > 0:
            random_idx = np.random.randint(len(self.history))
            time,uname,msg = self.history[random_idx]
            return msg, uname
        else:
            return '什么都没有', '虚空'


    async def _on_heartbeat(self, client: blivedm.BLiveClient, message: blivedm.HeartbeatMessage):
        #print(f'[{client.room_id}] 当前人气值：{message.popularity}')
        if self.flower.need_water():
            msg, uname = self.get_random_sentence_and_uname()
            self.flower.water(msg, uname)

    def record_start(self, uname):
        cyan('{self.prefix()}: 开始记录弹幕片段以组合成完整请求...结束并提问请输入"结束"')
        self.message_dict[uname] = {'is_recording': True, 'lst': []}

    def record_end(self, uname):
        if uname not in self.message_dict or len(self.message_dict[uname]['lst']) == 0:
            red('B话都没说，结束你🏇呢?')
        else:
            self.message_dict[uname]['is_recording'] = False
            request_text = ''.join(self.message_dict[uname]['lst'])
            cyan(f'{uname}: {request_text} \n生成中...')
            response_txt = self.ask(request_text, uname)
            cyan(f'{self.prefix()}: {response_txt}\n')
            say(response_txt)

    def record_list(self, uname):
        if uname in self.message_dict or len(self.message_dict[uname]['lst']) > 0:
            magenta(self.message_dict[uname]['lst'])
        else:
            red('B话都没说，列出你🏇呢?')

    def maybe_record(self, uname, msg):
        if uname in self.message_dict and self.message_dict[uname]['is_recording']:
            if msg == '记录':
                pass
            else:
                self.message_dict[uname]['lst'].append(msg)

    def ask(self, prompt, uname = None):
        if self.model_index in [1, 2]:
            if uname: 
                if uname not in self.dialogue_hist:
                    self.dialogue_hist[uname] = []
                # print(self.dialogue_hist[uname])
                dialogues = self.dialogue_hist[uname][-3:]
                small = True if self.model_index == 2 else False
                response_txt = send_rwkv_chat_dialogue(prompt, dialogues, small = small)
                self.dialogue_hist[uname].append((prompt, response_txt))
            else:
                print('已不再支持')
                # response_txt = send(prompt, rwkv = True)
        else:
            print('已不再支持')
            # response_txt, translated_prompt = send(prompt, trans=True, trans_prompt=True)
        return response_txt


    def prefix(self):
        if self.model_index == 1:
            return '升级完全版taku'
        if self.model_index == 2:
            return '升级但是余裕版taku'
        else:
            return '贫穷版taku'

    async def _on_danmaku(self, client: blivedm.BLiveClient, message: blivedm.DanmakuMessage):
        uname = message.uname
        msg = message.msg
        time = get_time_text(':')
        if len(msg) > 2:
            self.history.append((time,uname,msg))
        print(f'[{time}]{uname}:{msg}')
        say(message.msg)
        # 通过弹幕控制 gpt4all
        maybe_command_and_message = message.msg.split(' ')
        head = maybe_command_and_message[0]
        if len(maybe_command_and_message) > 1: # 用空格隔开的情况
            prompt = ' '.join(maybe_command_and_message[1:])
            if head in ['taku', '提问']:
                cyan('思考中...')
                response_txt = self.ask(prompt, uname)
                cyan(f'{self.prefix()}: {response_txt}\n')
                say(response_txt)
        else:
            if head == '记录':
                self.record_start(uname)
            elif head == '结束':
                self.record_end(uname)
            elif head == '列出':
                self.record_list(uname)
            elif head == '帮助':
                magenta(help_message)
            elif head == '历史':
                self.history_print()
            elif head == '保存':
                self.write_history()
            elif head == '花花':
                self.flower.show()
        self.maybe_record(uname, message.msg)
        # 判断是我的情况
        if uname == 'taku的交错电台':
            if head == '切换模型':
                self.model_index = (self.model_index + 1) % 3
                red(f'切换模型(只对taku的命令生效): {MODELS[self.model_index]}')

    async def _on_gift(self, client: blivedm.BLiveClient, message: blivedm.GiftMessage):
        text = f'谢谢{message.uname}赠送{message.num}个{message.gift_name}!'
        yellow(text)
        say(text)
        # print(f'[{client.room_id}] {message.uname} 赠送{message.gift_name}x{message.num}'
        # f' （{message.coin_type}瓜子x{message.total_coin}）')

    async def _on_buy_guard(self, client: blivedm.BLiveClient, message: blivedm.GuardBuyMessage):
        say('谢谢礼物!')
        print(f'[{client.room_id}] {message.username} 购买{message.gift_name}')

    async def _on_super_chat(self, client: blivedm.BLiveClient, message: blivedm.SuperChatMessage):
        print(f'[{client.room_id}] 醒目留言 ¥{message.price} {message.uname}：{message.message}')


if __name__ == '__main__':
    asyncio.run(main())
