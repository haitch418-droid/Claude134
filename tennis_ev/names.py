"""Match player names across sources.

Results files usually use "Alcaraz C." while Betfair uses "Carlos Alcaraz".
Both are reduced to (surname, first initial) and compared.
"""
from __future__ import annotations

import re
import unicodedata


def _clean(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace("-", " ").replace("'", "")
    s = re.sub(r"[^a-z. ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse(name: str) -> tuple[str, str]:
    """Return (surname, first initial) for either naming style."""
    s = _clean(name)
    tokens = s.split()
    if not tokens:
        return "", ""
    # "Alcaraz C." / "Auger Aliassime F." / "Ramos Vinolas A.": initials at the end
    if tokens[-1].endswith(".") or len(tokens[-1].rstrip(".")) == 1:
        initials = [t for t in tokens if t.endswith(".") or len(t.rstrip(".")) == 1]
        surname = [t for t in tokens if t not in initials]
        return " ".join(surname), initials[0][0]
    # "Carlos Alcaraz" / "Alex de Minaur": first name first
    if len(tokens) == 1:
        return tokens[0], ""
    return " ".join(tokens[1:]), tokens[0][0]


def key(name: str) -> str:
    surname, initial = parse(name)
    return f"{surname} {initial}".strip()


class NameMatcher:
    """Resolve names from any source to the names used in the results data."""

    def __init__(self, known_names, aliases: dict[str, str] | None = None):
        self.known = list(known_names)
        self.aliases = {_clean(k): v for k, v in (aliases or {}).items()}
        self._by_key: dict[str, list[str]] = {}
        self._parsed = {}
        for n in self.known:
            self._by_key.setdefault(key(n), []).append(n)
            self._parsed[n] = parse(n)

    def resolve(self, name: str) -> str | None:
        if name in self._parsed:
            return name
        cleaned = _clean(name)
        if cleaned in self.aliases:
            return self.aliases[cleaned]
        hits = self._by_key.get(key(name))
        if hits and len(hits) == 1:
            return hits[0]
        # Surname-suffix match: "Alex de Minaur" vs "De Minaur A."
        full = cleaned.replace(".", "")
        surname, initial = parse(name)
        cands = []
        for n, (s, i) in self._parsed.items():
            if not s:
                continue
            if (full.endswith(" " + s) or full == s or s.endswith(" " + surname) or s == surname) \
                    and (not initial or not i or i == initial):
                cands.append(n)
        if len(cands) == 1:
            return cands[0]
        return None
