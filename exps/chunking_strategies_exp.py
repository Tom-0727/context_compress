#!/usr/bin/env python3
"""衡量换行分块与句子分块两个简单策略的耗时差异。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.text_cleaning import clean_document_text


CHAR_LIMIT = 500  # 单个块最大字符数，两种策略保持一致
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "cache" / "tavily_exa_results.json"
OUTPUT_FILE = ROOT / "cache" / "chunking_strategies_results.json"


def load_inputs(path: Path) -> Dict[str, Dict]:
    """读取缓存文件内容并返回 results 字段。"""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    
    data = data.get("results", {})
    data = data.get("59", {})
    results = data.get("results", {})

    return results


def chunk_by_newline(text: str) -> List[str]:
    """方法A：基于换行的粗分块。"""
    return merge_segments(text.split("\n"), CHAR_LIMIT)


def chunk_by_sentence(text: str, sent_tokenize: Callable[[str], List[str]]) -> List[str]:
    """方法B：按句子拆分，再按阈值聚合。"""
    sentences = sent_tokenize(text)
    return merge_segments(sentences, CHAR_LIMIT)


def merge_segments(segments: Iterable[str], char_limit: int) -> List[str]:
    """按字符阈值累积基础分段，超出阈值立即落盘。"""
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
            chunks.append("\n".join(current))  # 切块：避免超过阈值
            current = [piece]
            current_len = piece_len
        else:
            current.append(piece)
            current_len = prospective
    if current:
        chunks.append("\n".join(current))
    return chunks


def ensure_sentence_tokenizer() -> Callable[[str], List[str]]:
    """确保句子分割器可用，缺少资源时尝试下载 punkt 与 punkt_tab。"""
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError as err:
        print("方法B需要 nltk 库，请先运行 `pip install nltk` 后再试。", file=sys.stderr)
        raise SystemExit(1) from err

    try:
        sent_tokenize("test sentence.")
        return sent_tokenize
    except LookupError:
        import nltk
        # punkt_tab 覆盖部分语言配置，两个资源一起补齐最保险。
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.download(resource)
            except Exception as download_err:  # pragma: no cover - 防御性退出
                print(
                    f"nltk 资源 `{resource}` 下载失败，请手动运行 "
                    f"`python -c \"import nltk; nltk.download('{resource}')\"`。",
                    file=sys.stderr,
                )
                raise SystemExit(1) from download_err

        try:
            sent_tokenize("test sentence.")
            return sent_tokenize
        except LookupError as err:  # pragma: no cover - 防御性退出
            print("nltk 资源仍然缺失，请确认 `punkt` 与 `punkt_tab` 已正确安装。", file=sys.stderr)
            raise SystemExit(1) from err



def main() -> None:
    """加载数据、执行两种分块策略并输出平均耗时和详细结果。"""
    if not INPUT_FILE.exists():
        print(f"未找到输入文件: {INPUT_FILE}", file=sys.stderr)
        raise SystemExit(1)

    inputs = load_inputs(INPUT_FILE)
    if not inputs:
        print("输入数据为空，无法运行实验。", file=sys.stderr)
        raise SystemExit(1)

    sent_tokenize = ensure_sentence_tokenizer()
    stats = {"count": 0, "time_a": 0.0, "time_b": 0.0, "chunks_a": 0, "chunks_b": 0}
    details: Dict[str, Dict] = {}

    for index, payload in enumerate(inputs):
        text = clean_document_text(payload).strip()
        if not text:
            print(f"跳过空文本 document_index={index}", file=sys.stderr)
            continue

        doc_key = payload.get("url") or f"document_{index}"
        start_a = time.perf_counter()
        chunks_a = chunk_by_newline(text)
        time_a_ms = (time.perf_counter() - start_a) * 1000

        start_b = time.perf_counter()
        chunks_b = chunk_by_sentence(text, sent_tokenize)
        time_b_ms = (time.perf_counter() - start_b) * 1000

        stats["count"] += 1
        stats["time_a"] += time_a_ms
        stats["time_b"] += time_b_ms
        stats["chunks_a"] += len(chunks_a)
        stats["chunks_b"] += len(chunks_b)

        details[doc_key] = {
            "source_index": index,
            "url": payload.get("url"),
            "query_id": payload.get("query_id"),
            "text_length": len(text),
            "method_a": {
                "time_ms": time_a_ms,
                "chunk_count": len(chunks_a),
                "chunks": chunks_a,
            },
            "method_b": {
                "time_ms": time_b_ms,
                "chunk_count": len(chunks_b),
                "chunks": chunks_b,
            },
        }

    if stats["count"] == 0:
        print("所有文档都为空，实验没有执行。", file=sys.stderr)
        raise SystemExit(1)

    avg_time_a = stats["time_a"] / stats["count"]
    avg_time_b = stats["time_b"] / stats["count"]
    avg_chunks_a = stats["chunks_a"] / stats["count"]
    avg_chunks_b = stats["chunks_b"] / stats["count"]

    print("实验完成：")
    print(f"- 方法A 平均耗时: {avg_time_a:.2f} ms, 平均块数量: {avg_chunks_a:.2f}")
    print(f"- 方法B 平均耗时: {avg_time_b:.2f} ms, 平均块数量: {avg_chunks_b:.2f}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_payload: Dict[str, object] = {
        "char_limit": CHAR_LIMIT,
        "document_count": stats["count"],
        "average_time_ms": {"method_a": avg_time_a, "method_b": avg_time_b},
        "average_chunks": {"method_a": avg_chunks_a, "method_b": avg_chunks_b},
        "details": details,
    }
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, ensure_ascii=False, indent=2)
    print(f"详细结果已写入 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
