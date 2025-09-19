import torch

from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout

from .pyramid_attention import PyramidAttention
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones


class PAM_SAKT(Module):
    def __init__(
        self,
        num_c,
        seq_len,
        emb_size,
        num_attn_heads,
        dropout,
        num_en=2,
        num_pyramid_levels=2,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
    ):
        super().__init__()
        self.model_name = "pam_sakt"
        self.emb_type = emb_type

        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(num_c * 2, emb_size)
            self.exercise_emb = Embedding(num_c, emb_size)
        self.position_emb = Embedding(seq_len, emb_size)

        block = Blocks(emb_size, num_attn_heads, dropout, num_pyramid_levels)
        self.blocks = get_clones(block, self.num_en)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)

    def base_emb(self, q, r, qry):
        x = q + self.num_c * r
        qshftemb = self.exercise_emb(qry)
        xemb = self.interaction_emb(x)

        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, q, r, qry, qtest=False):
        emb_type = self.emb_type
        if emb_type == "qid":
            qshftemb, xemb = self.base_emb(q, r, qry)
        else:
            raise NotImplementedError("PAM_SAKT currently supports emb_type='qid' only")

        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        if not qtest:
            return p
        else:
            return p, xemb


class Blocks(Module):
    def __init__(self, emb_size, num_attn_heads, dropout, num_pyramid_levels) -> None:
        super().__init__()

        self.attn = PyramidAttention(
            emb_size,
            num_attn_heads,
            dropout=dropout,
            num_levels=num_pyramid_levels,
        )
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        causal_mask = ut_mask(seq_len=k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb
