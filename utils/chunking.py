"""文本分块工具。"""

from __future__ import annotations

import sys
from typing import Callable, Iterable, List


CHAR_LIMIT = 500  # 单个块最大字符数


def chunk_documents(text: str) -> List[str]:
    """将文本按句子分割并按字符阈值聚合成块。

    [目标]: 将长文本切分为适合 LLM 处理的固定大小文本块。
    [对象]: 输入的原始文本字符串。
    [行为]: 使用 nltk 的句子分词器将文本拆分为句子，然后按 CHAR_LIMIT 阈值聚合句子。
    [约束]:
        - 每个块不超过 CHAR_LIMIT 字符（默认 500）
        - 句子不会被截断，完整句子会被分配到同一个块中
        - 需要 nltk 库及其 punkt/punkt_tab 资源

    Args:
        text: 待分块的文本内容

    Returns:
        分块后的文本列表

    Raises:
        SystemExit: 当 nltk 库缺失或资源下载失败时
    """
    sent_tokenize = ensure_sentence_tokenizer()
    sentences = sent_tokenize(text)
    return merge_segments(sentences, CHAR_LIMIT)


def merge_segments(segments: Iterable[str], char_limit: int) -> List[str]:
    """按字符阈值累积基础分段，超出阈值立即落盘。

    [目标]: 将小的文本片段合并成不超过字符限制的较大块。
    [对象]: 可迭代的文本片段序列。
    [行为]: 累积片段直到达到字符限制，然后创建新块；片段之间用换行符连接。
    [约束]:
        - 单个块的字符数不超过 char_limit
        - 空片段会被跳过
        - 片段之间用换行符 '\n' 连接

    Args:
        segments: 文本片段的可迭代对象
        char_limit: 单个块的最大字符数

    Returns:
        合并后的文本块列表
    """
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for raw_segment in segments:
        piece = raw_segment.strip()
        if not piece:
            continue

        piece_len = len(piece)
        prospective = piece_len if not current else current_len + 1 + piece_len

        if current and prospective > char_limit:
            chunks.append("\n".join(current))
            current = [piece]
            current_len = piece_len
        else:
            current.append(piece)
            current_len = prospective

    if current:
        chunks.append("\n".join(current))

    return chunks


def ensure_sentence_tokenizer() -> Callable[[str], List[str]]:
    """确保句子分割器可用，缺少资源时尝试下载 punkt 与 punkt_tab。

    [目标]: 提供可靠的句子分词功能，自动处理依赖下载。
    [对象]: nltk 的 sent_tokenize 函数。
    [行为]: 检查 nltk 是否安装及其资源是否可用；如果资源缺失则自动下载。
    [约束]:
        - 需要已安装 nltk 库
        - 需要能够访问 nltk 的数据下载服务
        - 下载失败时抛出 SystemExit

    Returns:
        nltk 的 sent_tokenize 函数

    Raises:
        SystemExit: 当 nltk 库缺失或资源下载失败时
    """
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError as err:
        print(
            "需要 nltk 库，请先运行 `pip install nltk` 后再试。",
            file=sys.stderr,
        )
        raise SystemExit(1) from err

    try:
        sent_tokenize("test sentence.")
        return sent_tokenize
    except LookupError:
        import nltk

        # punkt_tab 覆盖部分语言配置，两个资源一起补齐最保险
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.download(resource)
            except Exception as download_err:
                print(
                    f"nltk 资源 `{resource}` 下载失败，请手动运行 "
                    f"`python -c \"import nltk; nltk.download('{resource}')\"`。",
                    file=sys.stderr,
                )
                raise SystemExit(1) from download_err

        try:
            sent_tokenize("test sentence.")
            return sent_tokenize
        except LookupError as err:
            print(
                "nltk 资源仍然缺失，请确认 `punkt` 与 `punkt_tab` 已正确安装。",
                file=sys.stderr,
            )
            raise SystemExit(1) from err


__all__ = ["chunk_documents", "CHAR_LIMIT"]
