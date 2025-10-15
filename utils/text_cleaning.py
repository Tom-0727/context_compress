from __future__ import annotations

import re
from typing import Dict

from bs4 import BeautifulSoup
from ftfy import fix_text


_MD_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL_PATTERN = re.compile(r"https?://\S+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_document_text(payload: Dict[str, str]) -> str:
    """提取并清洗单条搜索结果文本。"""
    if not payload:
        return ""

    candidate = payload.get("content") or payload.get("snippet") or ""
    if not candidate:
        return ""

    normalized = fix_text(candidate)
    stripped = _strip_html(normalized)
    cleaned = _remove_markdown_noise(stripped)
    return cleaned


def _strip_html(raw_text: str) -> str:
    """移除残留的HTML标签并保留段落换行。"""
    if not raw_text:
        return ""
    try:
        soup = BeautifulSoup(raw_text, "lxml")
    except Exception:
        soup = BeautifulSoup(raw_text, "html.parser")
    return soup.get_text("\n", strip=True)


def _remove_markdown_noise(text: str) -> str:
    """去除Markdown标记、URL与多余空白。"""
    if not text:
        return ""

    text = _MD_IMAGE_PATTERN.sub(r"\1", text)
    text = _MD_LINK_PATTERN.sub(r"\1", text)
    text = _URL_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


__all__ = ["clean_document_text"]
