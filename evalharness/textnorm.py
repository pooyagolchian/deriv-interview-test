"""Single shared text normalizer used by retrieval, grounding, and rule checks.

Keeping tokenization in one place guarantees the stages never drift (e.g. retrieval and
grounding always agree on what a "token" is). Everything here is pure and deterministic.

Notes:
- Negation words (not, no, cannot, never, none, nor) are intentionally NOT stopwords, so
  "screenshots are not accepted" and "screenshots are accepted" do not look identical.
- Stemming is a conservative, dependency-free plural/suffix stripper — enough to unify
  withdrawal/withdrawals, review/reviews, statement/statements without an external library.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Set

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common function words removed for content comparison. Negations are deliberately absent.
STOPWORDS: frozenset = frozenset(
    """
    a an the and or but if then else when at by for in of on to up as is are was were be
    been being do does did doing have has had having i you he she it we they me him her them
    my your his its our their this that these those from with into onto over under again
    further once here there all any both each few more most other some such own same so than
    too very can will just would should could may might must about after before between during
    through because while also get got yours ours how what why who whom which where us then
    """.split()
)


@lru_cache(maxsize=8192)
def _stem(tok: str) -> str:
    """Deterministic conservative singularizer/suffix stripper."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"          # policies -> policy
    if len(tok) > 4 and tok.endswith(("sses", "shes", "ches", "xes", "zes")):
        return tok[:-2]                # statuses -> status, watches -> watch
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]                # withdrawals -> withdrawal
    return tok


def _split(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.casefold())


def raw_tokens(text: str) -> List[str]:
    """Ordered, stemmed tokens WITHOUT stopword removal."""
    return [_stem(t) for t in _split(text)]


def tokens(text: str) -> List[str]:
    """Ordered content tokens: casefold -> [a-z0-9]+ -> drop stopwords -> stem."""
    return [_stem(t) for t in _split(text) if t not in STOPWORDS]


def content_tokens(text: str) -> Set[str]:
    """Unique content tokens (the set form of :func:`tokens`)."""
    return set(tokens(text))


def normalize_spaces(text: str) -> str:
    """Casefold and collapse whitespace — for literal normalized-substring matching."""
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()
