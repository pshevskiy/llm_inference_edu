"""Development export graph for the already-verified T04 fixture, not an engine."""
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent

class TinyExport(torch.nn.Module):
    def __init__(self):
        super().__init__()
        fixture = json.loads((ROOT / 'fixtures/tiny-model.json').read_text())
        self.c = fixture['config']
        c = self.c
        assert (c['layers'], c['hidden_dim'], c['query_heads'], c['kv_heads'],
                c['head_dim'], c['ff_dim'], c['vocab_size']) == (2,32,4,2,8,64,64)
        weights = fixture['weights']
        self.register_buffer('embedding', torch.tensor(weights['embedding']).reshape(64,32))
        self.register_buffer('final_norm', torch.tensor(weights['final_norm']))
        for layer, values in enumerate(weights['layers']):
            for name, data in values.items():
                if data is None:
                    continue
                self.register_buffer(f'l{layer}_{name}', torch.tensor(data))
        self.register_buffer('head_map', torch.tensor([0,0,1,1]))
        self.register_buffer('frequency', c['rope_base'] ** (-2*torch.arange(4,dtype=torch.float32)/8))

    def norm(self, x, w):
        return x * torch.rsqrt((x*x).mean(-1, keepdim=True) + self.c['norm_epsilon']) * w

    def rotate(self, x, cos, sin):
        a, b = x[...,:4], x[...,4:]
        return torch.cat((a*cos-b*sin, b*cos+a*sin), -1)

    def forward(self, token_ids):
        # Shape remains symbolic through tracing; the exported input is bounded.
        t = token_ids.shape[0]
        positions = torch.arange(t, dtype=torch.float32)
        angles = positions[:,None,None] * self.frequency[None,None,:]
        cos, sin = torch.cos(angles), torch.sin(angles)
        allowed = positions[None,:] <= positions[:,None]
        x = torch.nn.functional.embedding(token_ids, self.embedding)
        results = [x]
        for layer in range(2):
            w = lambda name: getattr(self, f'l{layer}_{name}')
            an = self.norm(x, w('attention_norm'))
            q = (an @ w('query').reshape(32,32).T).reshape(t,4,8)
            k = (an @ w('key').reshape(16,32).T).reshape(t,2,8)
            v = (an @ w('value').reshape(16,32).T).reshape(t,2,8)
            qr, kr = self.rotate(q,cos,sin), self.rotate(k,cos,sin)
            expanded_k, expanded_v = kr[:,self.head_map,:], v[:,self.head_map,:]
            scores = qr.permute(1,0,2) @ expanded_k.permute(1,2,0) / (8 ** .5)
            scores = torch.where(allowed[None,:,:], scores, torch.full_like(scores, float('-inf')))
            p = torch.softmax(scores, dim=-1)
            context = (p @ expanded_v.permute(1,0,2)).permute(1,0,2).reshape(t,32)
            ao = context @ w('attention_output').reshape(32,32).T
            after = x + ao
            fn = self.norm(after, w('ffn_norm'))
            gate, up = fn @ w('gate').reshape(64,32).T, fn @ w('up').reshape(64,32).T
            activated = torch.nn.functional.silu(gate)
            product = activated * up
            fo = product @ w('down').reshape(32,64).T
            x = after + fo
            results.extend((an,q,k,v,qr,kr,context,ao,after,fn,gate,up,activated,product,fo,x))
        final = self.norm(x, self.final_norm)
        results.extend((final, final @ self.embedding.T))
        return tuple(results)
