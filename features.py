"""Feature extraction for the human-vs-synthetic prompt detector.

- heuristic_features(text): cheap stylistic features, no heavy dependencies.
- LMScorer: GPT-2 based features (perplexity and GLTR-style token-rank stats).
- all_features(text, scorer=None): both combined.
"""
from __future__ import annotations

import math
import re
import statistics

STOCK_PHRASES = [
    "delve", "comprehensive", "ensure that", "in today's", "as an expert",
    "step-by-step", "additionally", "furthermore", "tailored", "leverage",
    "you are a", "it is important to", "detailed",
]
INFORMAL_WORDS = {"pls", "plz", "u", "thx", "thanks", "idk", "lol", "gonna", "wanna", "ur", "gotta", "kinda", "tbh", "btw"}
# "thanks" is intentionally excluded below: it is common in polite human and synthetic text alike.
INFORMAL_WORDS.discard("thanks")

HEURISTIC_FEATURE_NAMES = [
    "length_words", "stock_phrase_count", "em_dash_rate", "colon_rate",
    "bullet_lines", "informal_count", "starts_lowercase", "no_end_punct",
    "type_token_ratio", "burstiness", "avg_word_len",
]
LM_FEATURE_NAMES = ["perplexity", "top10_frac", "top100_frac", "rank_std"]

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def heuristic_features(text: str) -> dict:
    text = text or ""
    stripped = text.strip()
    words = _words(stripped)
    lower_words = [w.lower() for w in words]
    n = len(words)
    lower = stripped.lower()

    stock = sum(lower.count(p) for p in STOCK_PHRASES)
    dashes = stripped.count("\u2014")
    colons = stripped.count(":")
    bullets = sum(1 for ln in stripped.splitlines() if _BULLET_RE.match(ln))
    informal = sum(1 for w in lower_words if w in INFORMAL_WORDS)

    first_alpha = next((c for c in stripped if c.isalpha()), "")
    starts_lower = int(bool(first_alpha) and first_alpha.islower())
    no_end = int(bool(stripped) and stripped[-1] not in ".!?\"')")

    ttr = len(set(lower_words)) / n if n else 0.0

    sentences = [s for s in _SENT_SPLIT_RE.split(stripped) if _words(s)]
    lengths = [len(_words(s)) for s in sentences]
    if len(lengths) >= 2 and statistics.mean(lengths) > 0:
        burst = statistics.pstdev(lengths) / statistics.mean(lengths)
    else:
        burst = 0.0

    avg_len = sum(len(w) for w in words) / n if n else 0.0

    return {
        "length_words": n,
        "stock_phrase_count": stock,
        "em_dash_rate": dashes / n if n else 0.0,
        "colon_rate": colons / n if n else 0.0,
        "bullet_lines": bullets,
        "informal_count": informal,
        "starts_lowercase": starts_lower,
        "no_end_punct": no_end,
        "type_token_ratio": ttr,
        "burstiness": burst,
        "avg_word_len": avg_len,
    }


class LMScorer:
    """GPT-2 scorer. Loads torch/transformers lazily so heuristics work without them."""

    def __init__(self, model_name: str = "gpt2", device: str | None = None, max_tokens: int = 512):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()
        self.max_tokens = max_tokens

    def _empty(self) -> dict:
        # Neutral-ish defaults for inputs too short to score (fewer than 2 tokens).
        return {"perplexity": 0.0, "top10_frac": 0.0, "top100_frac": 0.0, "rank_std": 0.0}

    def score(self, text: str) -> dict:
        torch = self.torch
        text = (text or "").strip()
        if not text:
            return self._empty()
        bos = self.tokenizer.bos_token_id
        ids = self.tokenizer.encode(text)[: self.max_tokens - 1]
        if len(ids) < 2:
            return self._empty()
        # Prepend <|endoftext|> so the first real token is also scored.
        input_ids = torch.tensor([[bos] + ids], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0, :-1]  # predictions for ids
        targets = input_ids[0, 1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # rank = 1 + number of tokens with strictly higher logit than the true token
        true_logit = logits.gather(1, targets.unsqueeze(1))
        ranks = (logits > true_logit).sum(dim=1).float() + 1.0

        ppl = math.exp(-tok_lp.mean().item())
        return {
            "perplexity": ppl,
            "top10_frac": (ranks <= 10).float().mean().item(),
            "top100_frac": (ranks <= 100).float().mean().item(),
            "rank_std": ranks.std(unbiased=False).item() if len(ranks) > 1 else 0.0,
        }


def all_features(text: str, scorer: LMScorer | None = None) -> dict:
    feats = heuristic_features(text)
    if scorer is not None:
        feats.update(scorer.score(text))
    return feats
