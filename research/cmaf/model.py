"""CMAF-Net: patch/token/frame attention, sample-wise missing-modality gates."""
import torch
from torch import nn
import torch.nn.functional as F

MODALITIES = ('image', 'text', 'audio')

class CMAFNet(nn.Module):
    def __init__(self, input_dims=(512, 768, 128), hidden=128, heads=4, classes=2, dropout=.15):
        super().__init__()
        assert hidden % heads == 0
        self.input_dims = tuple(input_dims)
        self.hidden = hidden
        self.classes = classes
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden)) for d in input_dims])
        self.cross = nn.ModuleDict({f'{i}_{j}': nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
                                  for i in range(3) for j in range(3) if i != j})
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.gate = nn.Sequential(nn.Linear(hidden * 3 + 3, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, classes))

    def forward(self, features, present=None, return_attention=False):
        """features: 3 tensors [B, length, respective input_dim]; present: bool [B,3]."""
        if len(features) != 3:
            raise ValueError('Expected three modality feature tensors')
        batch = features[0].shape[0]
        x = [self.proj[i](features[i]) for i in range(3)]
        if present is None:
            present = torch.ones(batch, 3, dtype=torch.bool, device=x[0].device)
        present = present.to(device=x[0].device, dtype=torch.bool)
        if present.shape != (batch, 3) or not present.any(dim=1).all():
            raise ValueError('Each sample must contain at least one modality')
        attended, weights = [], {}
        for i in range(3):
            contributions = []
            for j in range(3):
                if i == j:
                    continue
                output, attn = self.cross[f'{i}_{j}'](x[i], x[j], x[j], need_weights=return_attention,
                                                       average_attn_weights=True)
                pair_enabled = (present[:, i] & present[:, j]).float().view(batch, 1, 1)
                contributions.append(output * pair_enabled)
                if return_attention:
                    weights[f'{MODALITIES[i]}<-{MODALITIES[j]}'] = attn.detach()
            pair_count = torch.stack([ (present[:, i] & present[:, j]).float() for j in range(3) if j != i ], dim=1).sum(dim=1).clamp(min=1).view(batch,1,1)
            enriched = self.norms[i](x[i] + sum(contributions) / pair_count)
            attended.append(enriched.mean(dim=1) * present[:, i:i+1].float())
        raw = self.gate(torch.cat(attended + [present.float()], dim=-1))
        gates = F.softmax(raw.masked_fill(~present, torch.finfo(raw.dtype).min), dim=-1)
        fused = sum(gates[:, i:i+1] * attended[i] for i in range(3))
        logits = self.head(fused)
        if return_attention:
            return logits, gates, weights
        return logits, gates

class ConcatBaseline(nn.Module):
    def __init__(self, input_dims=(512,768,128), hidden=128, classes=2):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d,hidden) for d in input_dims])
        self.head = nn.Sequential(nn.Linear(hidden*3,hidden), nn.ReLU(), nn.Linear(hidden,classes))
    def forward(self, features, present=None):
        if present is None:
            present = torch.ones(features[0].size(0),3,device=features[0].device,dtype=torch.bool)
        embeddings = [self.proj[i](f.mean(1)) * present[:,i:i+1].float() for i,f in enumerate(features)]
        return self.head(torch.cat(embeddings,dim=-1))

def modality_drop(present, p=.25):
    """Per sample: with probability p, drop ONE among currently available modalities if >=2."""
    out = present.clone()
    for row in out:
        candidates = torch.nonzero(row, as_tuple=False).flatten()
        if len(candidates) >= 2 and torch.rand(()) < p:
            idx = candidates[torch.randint(len(candidates),(1,)).item()]
            row[idx] = False
    return out
