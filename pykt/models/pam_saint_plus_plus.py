import torch
import torch.nn as nn
from torch.nn import Dropout
import pandas as pd

from .pyramid_attention import PyramidAttention
from .utils import transformer_FFN, get_clones, ut_mask, pos_encode
from torch.nn import Embedding, Linear


device = "cpu" if not torch.cuda.is_available() else "cuda"


class PAM_SAINT_PLUS_PLUS(nn.Module):
    def __init__(
        self,
        num_q,
        num_c,
        seq_len,
        emb_size,
        num_attn_heads,
        dropout,
        n_blocks=1,
        num_pyramid_levels=2,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
    ):
        super().__init__()
        if num_q == num_c and num_q == 0:
            assert num_q != 0
        self.num_q = num_q
        self.num_c = num_c
        self.model_name = "pam_saint++"
        self.num_en = n_blocks
        self.num_de = n_blocks
        self.emb_type = emb_type

        self.embd_pos = nn.Embedding(seq_len, embedding_dim=emb_size)

        if emb_type.startswith("qid"):
            encoder_block = Encoder_block(
                emb_size,
                num_attn_heads,
                num_q,
                num_c,
                seq_len,
                dropout,
                num_pyramid_levels,
                emb_path=emb_path,
                pretrain_dim=pretrain_dim,
            )
            self.encoder = get_clones(encoder_block, self.num_en)

        decoder_block = Decoder_block(
            emb_size,
            2,
            num_attn_heads,
            seq_len,
            dropout,
            num_q,
            num_c,
            num_pyramid_levels,
        )
        self.decoder = get_clones(decoder_block, self.num_de)

        self.dropout = Dropout(dropout)
        self.out = nn.Linear(in_features=emb_size, out_features=1)

    def forward(self, in_ex, in_cat, in_res, qtest=False):
        raw_in_ex = in_ex
        raw_in_cat = in_cat
        emb_type = self.emb_type

        if self.num_q > 0:
            in_pos = pos_encode(in_ex.shape[1])
        else:
            in_pos = pos_encode(in_cat.shape[1])
        in_pos = self.embd_pos(in_pos)

        first_block = True
        for i in range(self.num_en):
            if i >= 1:
                first_block = False
            if emb_type == "qid":
                in_ex = self.encoder[i](in_ex, in_cat, in_pos, first_block=first_block)
            in_cat = in_ex

        start_token = torch.tensor([[0]]).repeat(in_res.shape[0], 1).to(device)
        in_res = torch.cat((start_token, in_res), dim=-1)

        first_block = True
        for i in range(self.num_de):
            if i >= 1:
                first_block = False
            in_res = self.decoder[i](raw_in_ex, raw_in_cat, in_res, in_pos, en_out=in_ex, first_block=first_block)

        res = self.out(self.dropout(in_res))
        res = torch.sigmoid(res).squeeze(-1)
        if not qtest:
            return res
        else:
            return res, in_res


class Encoder_block(nn.Module):
    def __init__(
        self,
        dim_model,
        heads_en,
        total_ex,
        total_cat,
        seq_len,
        dropout,
        num_pyramid_levels,
        emb_path="",
        pretrain_dim=768,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.emb_path = emb_path
        self.total_cat = total_cat
        self.total_ex = total_ex
        if total_ex > 0:
            if emb_path == "":
                self.embd_ex = nn.Embedding(total_ex, embedding_dim=dim_model)
            else:
                embs = pd.read_pickle(emb_path)
                self.exercise_embed = Embedding.from_pretrained(embs)
                self.linear = Linear(pretrain_dim, dim_model)

        if total_cat > 0:
            self.emb_cat = nn.Embedding(total_cat, embedding_dim=dim_model)

        self.multi_en = PyramidAttention(
            embed_dim=dim_model,
            num_heads=heads_en,
            dropout=dropout,
            num_levels=num_pyramid_levels,
        )
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = Dropout(dropout)

        self.ffn_en = transformer_FFN(dim_model, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, in_ex, in_cat, in_pos, first_block=True):
        if first_block:
            embs = []
            if self.total_ex > 0:
                if self.emb_path == "":
                    in_ex = self.embd_ex(in_ex)
                else:
                    in_ex = self.linear(self.exercise_embed(in_ex))
                embs.append(in_ex)
            if self.total_cat > 0:
                in_cat = self.emb_cat(in_cat)
                embs.append(in_cat)
            out = embs[0]
            for i in range(1, len(embs)):
                out += embs[i]
            out = out + in_pos
        else:
            out = in_ex

        out = out.permute(1, 0, 2)
        n, _, _ = out.shape
        out = self.layer_norm1(out)
        skip_out = out
        out, _ = self.multi_en(out, out, out, attn_mask=ut_mask(seq_len=n))
        out = self.dropout1(out)
        out = out + skip_out

        out = out.permute(1, 0, 2)
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout2(out)
        out = out + skip_out

        return out


class Decoder_block(nn.Module):
    def __init__(
        self,
        dim_model,
        total_res,
        heads_de,
        seq_len,
        dropout,
        num_q,
        num_c,
        num_pyramid_levels,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_q = num_q
        self.num_c = num_c
        self.embd_res = nn.Embedding(total_res + 1, embedding_dim=dim_model)
        self.embd_ex = nn.Embedding(num_q * 2 + 1, embedding_dim=dim_model)
        self.emb_cat = nn.Embedding(num_c * 2 + 1, embedding_dim=dim_model)

        self.multi_de1 = PyramidAttention(
            embed_dim=dim_model,
            num_heads=heads_de,
            dropout=dropout,
            num_levels=num_pyramid_levels,
        )
        self.multi_de2 = PyramidAttention(
            embed_dim=dim_model,
            num_heads=heads_de,
            dropout=dropout,
            num_levels=num_pyramid_levels,
        )
        self.ffn_en = transformer_FFN(dim_model, dropout)

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, in_ex, in_cat, in_res, in_pos, en_out, first_block=True):
        if first_block:
            in_in = self.embd_res(in_res)
            que_emb = self.embd_ex(in_ex + self.num_q * in_res)
            cat_emb = self.emb_cat(in_cat + self.num_c * in_res)
            out = in_in + que_emb + cat_emb + in_pos
        else:
            out = in_res

        out = out.permute(1, 0, 2)
        n, _, _ = out.shape

        out = self.layer_norm1(out)
        skip_out = out
        out, _ = self.multi_de1(out, out, out, attn_mask=ut_mask(seq_len=n))
        out = self.dropout1(out)
        out = skip_out + out

        en_out = en_out.permute(1, 0, 2)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, _ = self.multi_de2(out, en_out, en_out, attn_mask=ut_mask(seq_len=n))
        out = self.dropout2(out)
        out = out + skip_out

        out = out.permute(1, 0, 2)
        out = self.layer_norm3(out)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout3(out)
        out = out + skip_out

        return out
