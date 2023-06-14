# -*- coding: utf-8 -*-
import asyncio
import random

import blivedm
from taku import say, cyan, magenta, red, get_time_text
from gpt4all import send, send_dialogues

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
        self.style = 'rwkv'
        self.dialogue_hist = {}

    def history_print(self):
        for time,uname,msg in self.history[-20:]:
            magenta(f'[{time}]{uname}:{msg}')

    def write_history(self):
        time = get_time_text(need_date = True)
        f = open(f'./log/{time}.hist','w+')
        for time,uname,msg in self.history:
            f.write(f'[{time}]{uname}:{msg}\n')
        magenta(f'记录结束')
        f.close()
        if self.style == 'rwkv':
            f = open(f'./log/{time}.dialogue.hist','w+')
            for uname in self.dialogue_hist:
                f.write(f'## {uname}\n')
                for question, answer in self.dialogue_hist[uname]:
                    f.write(f'{question}: {answer}\n')
            f.close()
            magenta(f'对话记录结束')

    def append_msg(self, uname, msg):
        if uname not in self.message_dict:
            self.message_dict[uname] = []
        self.message_dict[uname].append(msg)

    # # 演示如何添加自定义回调
    # _CMD_CALLBACK_DICT = blivedm.BaseHandler._CMD_CALLBACK_DICT.copy()
    #
    # # 入场消息回调
    # async def __interact_word_callback(self, client: blivedm.BLiveClient, command: dict):
    #     print(f"[{client.room_id}] INTERACT_WORD: self_type={type(self).__name__}, room_id={client.room_id},"
    #           f" uname={command['data']['uname']}")
    # _CMD_CALLBACK_DICT['INTERACT_WORD'] = __interact_word_callback  # noqa

    async def _on_heartbeat(self, client: blivedm.BLiveClient, message: blivedm.HeartbeatMessage):
        #print(f'[{client.room_id}] 当前人气值：{message.popularity}')
        pass

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
        if self.style == 'rwkv':
            if uname: 
                if uname not in self.dialogue_hist:
                    self.dialogue_hist[uname] = []
                # print(self.dialogue_hist[uname])
                dialogues = self.dialogue_hist[uname][-3:]
                response_txt = send_dialogues(prompt, dialogues)
                self.dialogue_hist[uname].append((prompt, response_txt))
            else:
                response_txt = send(prompt, rwkv = True)
        else:
            response_txt, translated_prompt = send(prompt, trans=True, trans_prompt=True)
        return response_txt


    def prefix(self):
        if self.style == 'rwkv':
            return '升级版taku'
        else:
            return 'taku'

    async def _on_danmaku(self, client: blivedm.BLiveClient, message: blivedm.DanmakuMessage):
        uname = message.uname
        msg = message.msg
        time = get_time_text(':')
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
        self.maybe_record(uname, message.msg)
        # 判断是我的情况
        if uname == 'taku的交错电台':
            if head == '切换模型':
                if self.style == 'gpt4all':
                    self.style = 'rwkv'
                else:
                    self.style = 'gpt4all'
                red(f'切换模型(只对taku的命令生效): {self.style}')

    async def _on_gift(self, client: blivedm.BLiveClient, message: blivedm.GiftMessage):
        say('谢谢礼物!')
        print(f'[{client.room_id}] {message.uname} 赠送{message.gift_name}x{message.num}'
             f' （{message.coin_type}瓜子x{message.total_coin}）')

    async def _on_buy_guard(self, client: blivedm.BLiveClient, message: blivedm.GuardBuyMessage):
        say('谢谢礼物!')
        print(f'[{client.room_id}] {message.username} 购买{message.gift_name}')

    async def _on_super_chat(self, client: blivedm.BLiveClient, message: blivedm.SuperChatMessage):
        print(f'[{client.room_id}] 醒目留言 ¥{message.price} {message.uname}：{message.message}')


if __name__ == '__main__':
    asyncio.run(main())
