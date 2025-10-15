from __future__ import annotations

import re
from typing import Optional

import chardet
import trafilatura
from bs4 import BeautifulSoup
from ftfy import fix_text
from readability import Document

from metagpt.logs import logger


def clean_web_content(
    raw_content: Optional[str] = None, content: Optional[str] = None, url: Optional[str] = None
) -> str:
    """
    清洗网页内容的主函数

    Args:
        raw_content: Tavily API返回的raw_content字段（HTML格式）
        content: Tavily API返回的content字段（已处理的文本）
        url: 网页URL（可选）

    Returns:
        清洗后的纯文本内容。如果清洗后长度小于原始content，返回原始content
    """
    # 如果没有任何输入，返回空字符串
    if not raw_content and not content:
        return ""

    # 如果只有content没有raw_content，直接返回content
    if not raw_content and content:
        return fix_text(content)

    # 修复编码问题
    content_fixed = fix_text(content) if content else ""
    raw_fixed = _detect_and_fix_encoding(raw_content) if raw_content else ""

    # 从raw_content提取干净文本
    cleaned_text = _extract_clean_text(raw_fixed)

    # 长度检查：如果清洗后的内容比原始content短，使用原始content
    if content_fixed and cleaned_text:
        if len(cleaned_text.strip()) < len(content_fixed.strip()):
            return content_fixed

    # 如果没有成功清洗出内容，返回原始content
    if not cleaned_text and content_fixed:
        return content_fixed

    return cleaned_text or ""


def _detect_and_fix_encoding(raw_data: any) -> str:
    """检测并修复文本编码"""
    if isinstance(raw_data, bytes):
        enc = chardet.detect(raw_data).get("encoding") or "utf-8"
        text = raw_data.decode(enc, errors="ignore")
    else:
        text = str(raw_data)
    return fix_text(text)


def _extract_clean_text(html_text: str) -> str:
    """从HTML中提取干净的文本内容"""
    if not html_text:
        return ""

    len(html_text)

    # 优先使用trafilatura（速度快，质量高）
    try:
        result = trafilatura.extract(
            html_text,
            favor_precision=True,
            include_links=False,  # 不包含链接
            include_images=False,  # 不包含图片
            deduplicate=True,  # 去重
        )
        if result:
            cleaned = _clean_markdown_elements(result)
            if cleaned and len(cleaned.strip()) > 100:  # 确保有足够内容
                return cleaned
            else:
                logger.warning("Trafilatura extraction result too short or empty after cleaning")
    except Exception as e:
        logger.warning(f"Trafilatura extraction failed: {e}")

    # 备用方案：使用readability
    try:
        doc = Document(html_text)
        summary_html = doc.summary(html_partial=True)
        cleaned_html = _strip_html_noise(summary_html)
        soup = BeautifulSoup(cleaned_html, "lxml")
        text = soup.get_text("\n", strip=True)
        if text:
            cleaned = _clean_markdown_elements(text)
            return cleaned
        else:
            logger.warning("Readability extraction result empty after cleaning")
    except Exception as e:
        logger.warning(f"Readability extraction failed: {e}")

    # 最后方案：直接用BeautifulSoup提取
    try:
        cleaned_html = _strip_html_noise(html_text)
        soup = BeautifulSoup(cleaned_html, "lxml")
        text = soup.get_text("\n", strip=True)
        cleaned = _clean_markdown_elements(text)
        return cleaned
    except Exception as e:
        logger.error(f"All text extraction methods failed, last error: {e}")
        return ""


def _strip_html_noise(html_text: str) -> str:
    """去除HTML中的噪音元素"""
    soup = BeautifulSoup(html_text, "lxml")

    # 移除脚本、样式、广告等噪音元素
    noise_tags = ["script", "style", "noscript", "iframe", "svg", "form", "button", "input"]
    for tag in noise_tags:
        for element in soup.find_all(tag):
            element.decompose()

    # 移除可能的广告/导航容器（基于class和id）
    for element in soup.find_all(["div", "aside", "header", "footer", "nav"]):
        # 获取class和id属性
        class_str = " ".join(element.get("class", [])).lower()
        id_str = (element.get("id") or "").lower()

        # 检查是否包含广告/导航相关关键词
        noise_keywords = [
            "ad",
            "ads",
            "advert",
            "banner",
            "popup",
            "modal",
            "sidebar",
            "header",
            "footer",
            "nav",
            "menu",
            "breadcrumb",
            "social",
            "share",
            "comment",
            "related",
            "recommend",
        ]

        if any(keyword in class_str or keyword in id_str for keyword in noise_keywords):
            element.decompose()

    return str(soup)


def _clean_markdown_elements(text: str) -> str:
    """清理Markdown格式的链接和图片，保留纯文本"""
    if not text:
        return text

    # 清理图片 ![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # 清理链接 [text](url) -> text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # 清理HTML标签（如果还有残留的）
    text = re.sub(r"<[^>]+>", "", text)

    # 清理URL
    text = re.sub(r"https?://[^\s]+", "", text)

    # 清理多余的空白
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

    # 清理特殊字符和符号噪音
    text = re.sub(r"[|×╳✕✖⨯※]{2,}", "", text)  # 重复的分隔符
    text = re.sub(r"^\s*[|×╳✕✖⨯※]\s*|\s*[|×╳✕✖⨯※]\s*$", "", text, flags=re.MULTILINE)  # 行首行尾的分隔符

    return text.strip()
