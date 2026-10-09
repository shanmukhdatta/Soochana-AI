"""Conservative ticker extraction from the ACTUAL headline/post, not search query."""
from __future__ import annotations
import re


class TickerMatcher:
    def __init__(self, aliases: dict[str,list[str]]):
        self.aliases = aliases
        self.allowed = set(aliases)
        self.alias_patterns = {
            t: [re.compile(r'(?<![A-Za-z0-9])' + re.escape(alias) + r'(?![A-Za-z0-9])',re.I)
                for alias in values]
            for t,values in aliases.items()
        }

    def match(self, text: str) -> dict[str,str]:
        found: dict[str,str] = {}
        cashtags = {t.upper() for t in re.findall(r'(?<![\w])\$([A-Za-z][A-Za-z0-9.]{0,9})(?![\w])',text)}
        for ticker in sorted(cashtags & self.allowed):
            found[ticker] = 'cashtag'
        for ticker in sorted(self.allowed):
            if ticker not in found and re.search(r'(?<![A-Za-z0-9])' + re.escape(ticker) + r'(?![A-Za-z0-9])',text):
                found[ticker] = 'ticker_symbol'
        for ticker,patterns in self.alias_patterns.items():
            if ticker not in found and any(p.search(text) for p in patterns):
                found[ticker] = 'company_alias'
        return dict(sorted(found.items()))
