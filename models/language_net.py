import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import RobertaModel
from collections import OrderedDict
from typing import Tuple, Union
transformers.utils.logging.set_verbosity(50)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class TextModel(nn.Module):
    def __init__(self, fc_dim, text_dim=768):
        super(TextModel, self).__init__()
        self.text_model = RobertaModel.from_pretrained("roberta-base")
        
        for name, params in self.text_model.named_parameters():
            params.requires_grad = False

    def forward(self, word_tokens, word_masks):
    
        outputs = self.text_model(word_tokens, attention_mask=word_masks)
        last_hidden_states = outputs.last_hidden_state[:, 0, :]
        
        return last_hidden_states
        
class ClipTextModel(nn.Module):
    def __init__(self, arch, fc_dim, text_dim=1024):
        super(ClipTextModel, self).__init__()
        
        pretrained_path = "/research/rxtan/news_cluster_data/pretrained_clip_resnet50.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs4/mnt/data/rxtan/news_cluster_data/pretrained_clip_resnet50.pth"
        
        state_dict = torch.load(pretrained_path)
        
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.context_length = context_length
        
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self.arch = arch
        
        pretrained_dict = state_dict
        model_dict = self.state_dict()
            
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v
                
        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
        
        if '-fc' in self.arch:
            self.fc = nn.Linear(text_dim, fc_dim)
            
        if 'finetune' not in self.arch:
            for k, v in self.named_parameters():
                v.requires_grad = False
        
        '''for k, v in self.named_parameters():
            if ('fc' in k and 'transformer' not in k) or 'text_projection' in k:
                v.requires_grad = True
            else:
                v.requires_grad = False'''
        
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
        
    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, word_tokens):
        text_feats = self.encode_text(word_tokens)

        if '-fc' in self.arch:
            text_feats = self.fc(text_feats)
            
        if '-normalize' in self.arch:
            text_feats = F.normalize(text_feats, dim=-1)
        
        return text_feats
        
class ClipVitTextModel(nn.Module):
    def __init__(self, arch, fc_dim, text_dim=512):
        super(ClipVitTextModel, self).__init__()
        
        pretrained_path = "/research/rxtan/news_cluster_data/pretrained_clip_vitb32.pth"
        if not os.path.exists(pretrained_path):
            pretrained_path = "/net/ivcfs4/mnt/data/rxtan/news_cluster_data/pretrained_clip_vitb32.pth"
        
        state_dict = torch.load(pretrained_path)
        
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width * 32 // 64
        
        self.context_length = context_length
        
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self.arch = arch
        
        pretrained_dict = state_dict
        model_dict = self.state_dict()
            
        updated_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                updated_pretrained_dict[k] = v
                
        model_dict.update(updated_pretrained_dict)
        self.load_state_dict(model_dict)
        
        if '-fc' in self.arch:
            self.fc = nn.Linear(text_dim, fc_dim)
        
        for k, v in self.named_parameters():
            if ('fc' in k and 'transformer' not in k) or 'text_projection' in k:
                v.requires_grad = True
            else:
                v.requires_grad = False
        
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
        
    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, word_tokens):
        text_feats = self.encode_text(word_tokens)

        if '-fc' in self.arch:
            text_feats = self.fc(text_feats)
            
        if '-normalize' in self.arch:
            text_feats = F.normalize(text_feats, dim=-1)
        
        return text_feats