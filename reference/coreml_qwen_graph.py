"""Development-only one-token Qwen export graph with explicit persistent KV state."""
import json
from pathlib import Path
import torch
from safetensors import safe_open

class QwenStateExport(torch.nn.Module):
    def __init__(self, directory: Path, state_dtype=torch.float16):
        super().__init__()
        c = json.loads((directory/'config.json').read_text())
        assert (c['hidden_size'],c['intermediate_size'],c['num_hidden_layers'],
                c['num_attention_heads'],c['num_key_value_heads'],c['vocab_size']) == (896,4864,24,14,2,151936)
        self.eps = c['rms_norm_eps']
        self.capacity = 32
        self.split_states = False
        self.tuple_outputs = False
        # safe_open maps verified BF16 tensors; expand/register one at a time.
        with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as tensors:
            self.register_buffer('embedding',tensors.get_tensor('model.embed_tokens.weight').float())
            self.register_buffer('final_norm',tensors.get_tensor('model.norm.weight').float())
            for layer in range(24):
                for local, source in [
                    ('attention_norm','input_layernorm.weight'),('ffn_norm','post_attention_layernorm.weight'),
                    ('query','self_attn.q_proj.weight'),('key','self_attn.k_proj.weight'),
                    ('value','self_attn.v_proj.weight'),('output','self_attn.o_proj.weight'),
                    ('query_bias','self_attn.q_proj.bias'),('key_bias','self_attn.k_proj.bias'),
                    ('value_bias','self_attn.v_proj.bias'),('gate','mlp.gate_proj.weight'),
                    ('up','mlp.up_proj.weight'),('down','mlp.down_proj.weight')]:
                    self.register_buffer(f'l{layer}_{local}',tensors.get_tensor(f'model.layers.{layer}.{source}').float())
        self.register_buffer('frequency',c['rope_theta'] ** (-2*torch.arange(32,dtype=torch.float32)/64))
        self.register_buffer('head_map',torch.arange(14)//7)
        self.register_buffer('slots',torch.arange(32,dtype=torch.int32))
        for layer in range(24):
            for kind in ('k','v'):
                self.register_buffer(f'{kind}{layer}',torch.zeros((2,32,64),dtype=state_dtype))

    def reset(self, state_dtype=None):
        for layer in range(24):
            for kind in ('k','v'):
                name = f'{kind}{layer}'
                old = getattr(self,name)
                setattr(self,name,torch.zeros_like(old,dtype=state_dtype or old.dtype))
                if self.split_states:
                    low = f'{kind}l{layer}'
                    setattr(self,low,torch.zeros_like(getattr(self,low)))

    def enable_split_state(self):
        self.reset(torch.float16)
        for layer in range(24):
            for kind in ('k','v'):
                self.register_buffer(f'{kind}l{layer}',torch.zeros((2,32,64),dtype=torch.float16))
        self.split_states = True

    def norm(self,x,w):
        return x * torch.rsqrt((x*x).mean(-1,keepdim=True)+self.eps) * w

    def rotate(self,x,cos,sin):
        a,b=x[...,:32],x[...,32:]
        return torch.cat((a*cos-b*sin,b*cos+a*sin),-1)

    def forward(self, token_ids, position):
        x = torch.nn.functional.embedding(token_ids,self.embedding)
        outputs = {'embedding':x}
        angle = position.to(torch.float32).reshape(1,1,1) * self.frequency.reshape(1,1,32)
        cos,sin = torch.cos(angle),torch.sin(angle)
        # Fixed capacity with explicit active-prefix masking; no dynamic lengths.
        mask = self.slots <= position[0]
        for layer in range(24):
            w = lambda name: getattr(self,f'l{layer}_{name}')
            an=self.norm(x,w('attention_norm'))
            q=torch.nn.functional.linear(an,w('query'),w('query_bias')).reshape(14,1,64)
            k=torch.nn.functional.linear(an,w('key'),w('key_bias')).reshape(2,1,64)
            v=torch.nn.functional.linear(an,w('value'),w('value_bias')).reshape(2,1,64)
            qr,kr=self.rotate(q,cos,sin),self.rotate(k,cos,sin)
            kc,vc=getattr(self,f'k{layer}'),getattr(self,f'v{layer}')
            kc[:,position[0]:position[0]+1,:]=kr.to(kc.dtype)
            vc[:,position[0]:position[0]+1,:]=v.to(vc.dtype)
            if self.split_states:
                kl,vl=getattr(self,f'kl{layer}'),getattr(self,f'vl{layer}')
                kl[:,position[0]:position[0]+1,:]=(kr-kr.half().float()).half()
                vl[:,position[0]:position[0]+1,:]=(v-v.half().float()).half()
                expanded_k=(kc.float()+kl.float())[self.head_map]
                expanded_v=(vc.float()+vl.float())[self.head_map]
            else:
                expanded_k,expanded_v=kc[self.head_map].float(),vc[self.head_map].float()
            scores=(qr @ expanded_k.transpose(-2,-1)) / 8
            probabilities=torch.softmax(torch.where(mask.reshape(1,1,32),scores,
                                    torch.full_like(scores,float('-inf'))),dim=-1)
            context=(probabilities @ expanded_v).transpose(0,1).reshape(1,896)
            ao=torch.nn.functional.linear(context,w('output'))
            after=x+ao
            fn=self.norm(after,w('ffn_norm'))
            gate,up=torch.nn.functional.linear(fn,w('gate')),torch.nn.functional.linear(fn,w('up'))
            product=torch.nn.functional.silu(gate)*up
            fo=torch.nn.functional.linear(product,w('down')); x=after+fo
            if layer in (0,23):
                # Same names/layouts as the independent Transformers hooks.
                named=dict(attention_norm=an,query=q.transpose(0,1).reshape(1,-1),
                    key=k.transpose(0,1).reshape(1,-1),value=v.transpose(0,1).reshape(1,-1),
                    context=context,attention_output=ao,ffn_norm=fn,gate=gate,up=up,
                    product=product,ffn_output=fo,output=x)
                outputs.update({f'layer_{layer}.{name}':value for name,value in named.items()})
            else:
                outputs[f'layer_{layer}.output']=x
        final=self.norm(x,self.final_norm)
        outputs['final_norm']=final
        outputs['logits']=torch.nn.functional.linear(final,self.embedding)
        return tuple(outputs.values()) if self.tuple_outputs else outputs
