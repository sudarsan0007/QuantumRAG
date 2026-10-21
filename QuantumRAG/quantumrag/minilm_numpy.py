"""Dependency-free inference for sentence-transformers/all-MiniLM-L6-v2.

This module runs the original MiniLM weights (model.safetensors, vocab.txt)
with NumPy only, reproducing what Sentence-Transformers does for this model:
BERT uncased WordPiece tokenisation, a 6-layer BERT encoder, mean pooling
over non-padding tokens and L2 normalisation. It exists so the benchmark can
use the real MiniLM model on machines where PyTorch or the Sentence-Transformers
package cannot be installed. Point QRAG_MINILM_DIR at a folder holding the
model files (for example a Hugging Face snapshot folder).
"""
from __future__ import annotations

import json
import struct
import unicodedata
from pathlib import Path

import numpy as np
from scipy.special import erf

_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64}


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        arr = np.frombuffer(raw[base + s: base + e], dtype=_DTYPES[meta["dtype"]])
        out[name] = arr.reshape(meta["shape"]).astype(np.float32)
    return out


# --------------------------------------------------------------------- tokenizer
def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class BertTokenizer:
    """BERT uncased basic + WordPiece tokenizer (matches HF BertTokenizer)."""

    def __init__(self, vocab_path: Path, max_len: int = 256):
        tokens = Path(vocab_path).read_text(encoding="utf-8").split("\n")
        if tokens and tokens[-1] == "":
            tokens = tokens[:-1]
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.max_len = max_len
        self.cls, self.sep = self.vocab["[CLS]"], self.vocab["[SEP]"]
        self.unk, self.pad = self.vocab["[UNK]"], self.vocab["[PAD]"]

    def _basic(self, text: str) -> list[str]:
        chars = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"):
                continue
            if ch in " \t\n\r" or unicodedata.category(ch) == "Zs":
                chars.append(" ")
            elif _is_cjk(cp):
                chars.append(f" {ch} ")
            else:
                chars.append(ch)
        words = []
        for tok in "".join(chars).strip().split():
            tok = unicodedata.normalize("NFD", tok.lower())
            tok = "".join(c for c in tok if unicodedata.category(c) != "Mn")
            cur = ""
            for c in tok:
                if _is_punct(c):
                    if cur:
                        words.append(cur)
                        cur = ""
                    words.append(c)
                else:
                    cur += c
            if cur:
                words.append(cur)
        return words

    def _wordpiece(self, word: str) -> list[int]:
        if len(word) > 100:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk]
            ids.append(found)
            start = end
        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        for w in self._basic(text):
            ids.extend(self._wordpiece(w))
        return [self.cls] + ids[: self.max_len - 2] + [self.sep]


# ----------------------------------------------------------------------- encoder
def _layer_norm(x, g, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


class MiniLMNumpy:
    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        self.h = cfg["hidden_size"]
        self.heads = cfg["num_attention_heads"]
        self.layers = cfg["num_hidden_layers"]
        self.eps = cfg.get("layer_norm_eps", 1e-12)
        max_len = 256
        sb = d / "sentence_bert_config.json"
        if sb.exists():
            max_len = json.loads(sb.read_text()).get("max_seq_length", 256)
        self.tok = BertTokenizer(d / "vocab.txt", max_len=max_len)
        w = load_safetensors(d / "model.safetensors")
        self.w = w  # keys may carry a "bert." or "0.auto_model." prefix
        self.prefix = ""
        for p in ("", "bert.", "0.auto_model."):
            if p + "embeddings.word_embeddings.weight" in w:
                self.prefix = p
                break
        self.dim = self.h

    def _p(self, name):
        return self.w[self.prefix + name]

    def _forward(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        B, T = ids.shape
        x = (self._p("embeddings.word_embeddings.weight")[ids]
             + self._p("embeddings.position_embeddings.weight")[np.arange(T)][None]
             + self._p("embeddings.token_type_embeddings.weight")[0][None, None])
        x = _layer_norm(x, self._p("embeddings.LayerNorm.weight"), self._p("embeddings.LayerNorm.bias"), self.eps)
        hd = self.h // self.heads
        bias = (1.0 - mask[:, None, None, :].astype(np.float32)) * -1e9
        for i in range(self.layers):
            pre = f"encoder.layer.{i}."

            def lin(t, n):
                return t @ self._p(pre + n + ".weight").T + self._p(pre + n + ".bias")

            def split(t):
                return t.reshape(B, T, self.heads, hd).transpose(0, 2, 1, 3)

            q = split(lin(x, "attention.self.query"))
            k = split(lin(x, "attention.self.key"))
            v = split(lin(x, "attention.self.value"))
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + bias
            s = s - s.max(-1, keepdims=True)
            a = np.exp(s)
            a /= a.sum(-1, keepdims=True)
            ctx = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, self.h)
            x = _layer_norm(x + lin(ctx, "attention.output.dense"),
                            self._p(pre + "attention.output.LayerNorm.weight"),
                            self._p(pre + "attention.output.LayerNorm.bias"), self.eps)
            ff = _gelu(lin(x, "intermediate.dense"))
            x = _layer_norm(x + lin(ff, "output.dense"),
                            self._p(pre + "output.LayerNorm.weight"),
                            self._p(pre + "output.LayerNorm.bias"), self.eps)
        m = mask[..., None].astype(np.float32)
        return (x * m).sum(1) / np.clip(m.sum(1), 1e-9, None)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        enc = [self.tok.encode(t) for t in texts]
        order = np.argsort([len(e) for e in enc])  # length bucketing for speed
        out = np.zeros((len(texts), self.h), dtype=np.float32)
        for s in range(0, len(texts), batch_size):
            idx = order[s: s + batch_size]
            T = max(len(enc[i]) for i in idx)
            ids = np.full((len(idx), T), self.tok.pad, dtype=np.int64)
            mask = np.zeros((len(idx), T), dtype=np.int64)
            for r, i in enumerate(idx):
                ids[r, : len(enc[i])] = enc[i]
                mask[r, : len(enc[i])] = 1
            out[idx] = self._forward(ids, mask)
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
