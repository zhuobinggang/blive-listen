import numpy as np
import torch
from torch.nn import functional as F

END_OF_TEXT = 0

class PIPELINE_ARGS():
    def __init__(self, temperature=1.2, top_p=0.5, top_k=0, alpha_frequency=0.4, alpha_presence=0.4, token_ban=[], token_stop=[END_OF_TEXT], chunk_len=256):
        # def __init__(self, temperature=1.0, top_p=0.85, top_k=0, alpha_frequency=0.2, alpha_presence=0.2, token_ban=[], token_stop=[], chunk_len=256):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alpha_frequency = alpha_frequency # Frequency Penalty (as in GPT-3)
        self.alpha_presence = alpha_presence # Presence Penalty (as in GPT-3)
        self.token_ban = token_ban # ban the generation of some tokens
        self.token_stop = token_stop # stop generation whenever you see any token here
        self.chunk_len = chunk_len # split input into chunks to save VRAM (shorter -> slower)

class PIPELINE():
    def __init__(self, model, tokenizer, opter, stop=f"\nBob"):
        self.model = model
        self.stop = stop
        self.prefix = 'Alice:'
        self.tokenizer = tokenizer
        self.opter = opter


    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def encode(self, x):
        return self.tokenizer(x, return_tensors='pt')['input_ids'].tolist()[0]

    def decode(self, x):
        return self.tokenizer.decode(x)

    # logits: 50277
    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        probs = F.softmax(logits.float(), dim=-1) #
        top_k = int(top_k)
        if probs.device == torch.device('cpu'):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)

    @torch.no_grad()
    def generate(self, ctx, token_count=100, args=PIPELINE_ARGS(), callback=None, state=None):
        all_tokens = []
        out_last = 0
        out_str = ''
        occurrence = {}
        for i in range(token_count):
            # forward & adjust prob.
            # 因为会保留state，所以chunk结束之后重新开始也没关系的样子
            tokens = self.encode(ctx) if i == 0 else [token] # 一开始encode ctx以后就[token]
            # tokens: [ids]
            while len(tokens) > 0:
                # chunk_len处截断，取hn
                # out, state = self.model.forward(tokens[:args.chunk_len], state)
                out = self.model(input_ids = torch.LongTensor([tokens[:args.chunk_len]]), state = state, use_cache = True)
                state = out.state
                out = out.logits[0, -1]
                tokens = tokens[args.chunk_len:]

            # 感觉没什么用
            for n in args.token_ban: # 不输出0
                out[n] = -float('inf')
            for n in occurrence:
                out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

            # sampler
            token = self.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            if token in args.token_stop:
                break
            all_tokens += [token]
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            # output
            tmp = self.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                if callback:
                    callback(tmp)
                out_str += tmp
                out_last = i + 1
                if self.stop in out_str:
                    # print(f'切断: {out_str}')
                    out_str = out_str.split(self.stop)[0]
                    break
        out_str = out_str.replace(self.prefix, '').strip()
        return out_str

    def finetune(self, ctx, answer, args=PIPELINE_ARGS(), callback=None, state=None):
        all_tokens = []
        out_last = 0
        out_str = ''
        occurrence = {}
        tokens = self.encode(ctx) # 一开始encode ctx
        # 获取state, 无backprop
        with torch.no_grad():
            while len(tokens) > 0:
                # chunk_len处截断，取hn
                # out, state = self.model.forward(tokens[:args.chunk_len], state)
                out = self.model(input_ids = torch.LongTensor([tokens[:args.chunk_len]]), state = state, use_cache = True)
                state = [item.clone().detach() for item in out.state] # NOTE: 试试能不能
                out = out.logits[0, -1]
                tokens = tokens[args.chunk_len:]
        # 计算loss
        label_ids = self.tokenizer(answer, return_tensors='pt')['input_ids']
        assert state is not None
        out = self.model(input_ids = label_ids, labels = label_ids, state = state, use_cache = True)
        self.opter.zero_grad()
        out.loss.backward()
        self.opter.step()
        return out.loss.item()


