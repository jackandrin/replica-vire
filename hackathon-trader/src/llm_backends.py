"""LLM backends for src/sentiment_agent.py, all behind the same tiny interface:
    complete_batch(prompts: list[str]) -> list[str]     (one raw completion per prompt, same order)
    embed(texts: list[str]) -> np.ndarray               (L2-normalized, shape [n, dim])
cortex = Snowflake Cortex (live server), hf = local transformers on a GPU (HPC backfill), fake = tests.
Heavy deps (snowflake, torch, transformers) are imported lazily.
Usage: from src.llm_backends import make_backend; be = make_backend("cortex", cfg()["sentiment"], schema)"""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

from .common import env, get_logger

log = get_logger("llm_backends")


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sf_literal(v) -> str:
    """Python value -> Snowflake OBJECT/ARRAY constant ({'k': 'v'}), as COMPLETE expects for options."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{sf_literal(str(k))}: {sf_literal(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(sf_literal(x) for x in v) + "]"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


class CortexBackend:
    """One SQL statement per batch: prompts go into a temp table, COMPLETE runs over all rows in parallel."""
    name = "cortex"

    def __init__(self, model: str, embed_model: str, schema: dict | None = None, max_tokens: int = 400):
        import snowflake.connector
        self.model_name, self.embed_model, self.schema, self.max_tokens = model, embed_model, schema, max_tokens
        self.embed_tag = f"cortex:{embed_model}"
        self.use_schema = schema is not None
        self.tokens = 0
        self.conn = snowflake.connector.connect(
            account=env("SNOWFLAKE_ACCOUNT"), user=env("SNOWFLAKE_USER"), password=env("SNOWFLAKE_PASSWORD"),
            warehouse=env("SNOWFLAKE_WAREHOUSE"), database=env("SNOWFLAKE_DATABASE"),
            schema=env("SNOWFLAKE_SCHEMA"), role=env("SNOWFLAKE_ROLE", False) or None)

    def _load(self, cur, table: str, texts: list) -> None:
        cur.execute(f"CREATE OR REPLACE TEMPORARY TABLE {table} (id NUMBER, txt STRING)")
        rows = [(i, t) for i, t in enumerate(texts)]
        for j in range(0, len(rows), 500):  # keep each multi-row INSERT a reasonable size
            cur.executemany(f"INSERT INTO {table} (id, txt) VALUES (%s, %s)", rows[j:j + 500])

    def _options(self) -> str:
        o = {"temperature": 0, "max_tokens": self.max_tokens}
        if self.use_schema:
            o["response_format"] = {"type": "json", "schema": self.schema}
        return sf_literal(o)

    @staticmethod
    def _unwrap(raw):
        """COMPLETE with options returns {"choices":[{"messages": ...}], "usage": {...}}."""
        if raw is None:
            return "", 0
        try:
            r = json.loads(raw) if isinstance(raw, str) else raw
            msg = r["choices"][0]["messages"]
            return (msg if isinstance(msg, str) else json.dumps(msg)), int(r.get("usage", {}).get("total_tokens", 0))
        except (ValueError, KeyError, IndexError, TypeError):
            return str(raw), 0

    def complete_batch(self, prompts: list) -> list:
        if not prompts:
            return []
        cur = self.conn.cursor()
        self._load(cur, "sa_prompts", prompts)
        sql = ("SELECT id, SNOWFLAKE.CORTEX.COMPLETE(%s, [{'role': 'user', 'content': txt}], {opts}) "
               "FROM sa_prompts ORDER BY id")
        try:
            cur.execute(sql.replace("{opts}", self._options()), (self.model_name,))
        except Exception as e:  # response_format unsupported for this model/account -> rely on parsing
            if not self.use_schema:
                raise
            log.warning("COMPLETE with response_format failed (%s); retrying without it", str(e)[:200])
            self.use_schema = False
            cur.execute(sql.replace("{opts}", self._options()), (self.model_name,))
        out = dict(cur.fetchall())
        res = []
        for i in range(len(prompts)):
            txt, n = self._unwrap(out.get(i))
            self.tokens += n
            res.append(txt)
        return res

    def embed(self, texts: list) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        cur = self.conn.cursor()
        self._load(cur, "sa_embed", texts)
        cur.execute("SELECT id, SNOWFLAKE.CORTEX.EMBED_TEXT_768(%s, txt)::ARRAY FROM sa_embed ORDER BY id",
                    (self.embed_model,))
        out = {int(i): (json.loads(v) if isinstance(v, str) else v) for i, v in cur.fetchall()}
        return _norm(np.array([out[i] for i in range(len(texts))], dtype=np.float32))


class HFBackend:
    """Local transformers: batched greedy generation with the chat template (bf16 on CUDA, left padding).
    Works offline from a pre-populated HF cache (HF_HOME, HF_HUB_OFFLINE=1)."""
    name = "hf"

    def __init__(self, model: str, embed_model: str, batch_size: int = 16, max_tokens: int = 400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.model_name, self.embed_model_name = torch, model, embed_model
        self.embed_tag = f"hf:{embed_model}"
        self.batch_size, self.max_tokens = batch_size, max_tokens
        self.local = os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True")
        cuda = torch.cuda.is_available()
        self.device = "cuda" if cuda else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model, local_files_only=self.local)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16 if cuda else torch.float32,
            device_map="auto" if cuda else None, local_files_only=self.local).eval()
        self._st = None
        log.info("loaded %s on %s", model, self.device)

    def _generate(self, prompts: list) -> list:
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.inference_mode():
            gen = self.model.generate(**enc, max_new_tokens=self.max_tokens, do_sample=False, temperature=None,
                                      top_p=None, pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def complete_batch(self, prompts: list) -> list:
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths -> less padding
        out, i, bs = [None] * len(prompts), 0, self.batch_size
        while i < len(order):
            idx = order[i:i + bs]
            try:
                res = self._generate([prompts[j] for j in idx])
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                log.warning("CUDA OOM; micro-batch size -> %d", bs)
                continue
            for j, r in zip(idx, res):
                out[j] = r
            i += len(idx)
        return out

    def embed(self, texts: list) -> np.ndarray:
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.embed_model_name, device=self.device,
                                           local_files_only=self.local)
        return _norm(self._st.encode(list(texts), batch_size=64, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False))


class FakeBackend:
    """Deterministic stand-in for tests: hashed bag-of-words embeddings, keyword-driven JSON answers."""
    name, model_name, embed_tag = "fake", "fake-v0", "fake"

    def __init__(self, dim: int = 64):
        self.dim, self.calls = dim, []  # calls = list of prompt batches, for assertions

    def complete_batch(self, prompts: list) -> list:
        self.calls.append(list(prompts))
        out = []
        for p in prompts:
            new = p.split("NEW DOCUMENT", 1)[-1].lower()
            up, down = new.count("beat") + new.count("raise"), new.count("miss") + new.count("cut")
            d = 0.0 if up == down else (0.5 if up > down else -0.5)
            m = re.search(r"level (\d+(?:\.\d+)?)/100", p)
            prior = float(m.group(1)) if m else 50.0
            out.append(json.dumps({
                "direction": "improving" if d > 0 else "deteriorating" if d < 0 else "stable", "delta": d,
                "level": prior + 20 * d, "novelty": 0.8, "materiality": 0.5 + abs(d), "horizon": "days",
                "drivers": ["beat" if d > 0 else "miss" if d < 0 else "none"],
                "updated_summary": f"fake view after {'up' if d > 0 else 'down' if d < 0 else 'flat'} news"}))
        return out

    def embed(self, texts: list) -> np.ndarray:
        x = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", str(t).lower()):
                x[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        x[:, 0] += 1e-3  # never an all-zero row
        return _norm(x)


def make_backend(kind: str, sc: dict, schema: dict | None = None, hf_batch: int = 16):
    """sc = config.yaml `sentiment:` block."""
    if kind == "cortex":
        return CortexBackend(sc["cortex_model"], sc["cortex_embed_model"], schema)
    if kind == "hf":
        return HFBackend(sc["hf_model"], sc["hf_embed_model"], batch_size=hf_batch)
    if kind == "fake":
        return FakeBackend()
    raise ValueError(f"unknown backend {kind}")
