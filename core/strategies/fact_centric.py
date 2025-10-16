"""策略A: 基于事实提取的压缩策略。"""

import time
import json
from pathlib import Path
from typing import List, Tuple

from core.strategies.base import CompressionStrategy
from core.types import FactObject
from utils.llm_client import call_llm


class FactCentricStrategy(CompressionStrategy):
    """事实中心压缩策略：从文本块中提取结构化事实并保持原文追溯能力。"""

    def __init__(self, verbose: bool = False) -> None:
        """初始化策略实例。

        Args:
            verbose: 是否打印耗时信息
        """
        super().__init__()
        self.verbose = verbose

    def _init_knowledge_base(self) -> List[FactObject]:
        """初始化空的事实列表。"""
        return []

    def process(
        self,
        query: str,
        search_results: List[Tuple[str, str]],
    ) -> None:
        """对搜索结果进行 chunking，提取事实并累积到知识库。"""


        start_time = time.time()
        chunk_pairs = self._chunk_and_store(search_results)
        new_facts = self._extract_facts(query, chunk_pairs)
        self.knowledge_base.extend(new_facts)

        if self.verbose:
            elapsed = time.time() - start_time
            print(f"⏱️  process() 耗时: {elapsed:.2f}s")

    def get_checklist_context(self) -> str:
        """返回事实对象的 JSON 字符串表示。"""
        if not self.knowledge_base:
            return "[]"

        # 将 FactObject 列表转换为字典列表，添加 fact_id
        facts_data = []
        for i, fact in enumerate(self.knowledge_base):
            facts_data.append({
                "fact_id": f"fact_{i}",
                "summary": fact.summary,
                "source_url": fact.source_url,
            })

        return json.dumps(facts_data, ensure_ascii=False, indent=2)

    def reconstruct_report_context(
        self,
        relevant_items: List[str] | None = None,
    ) -> str:
        """根据选中的事实 IDs 从 chunk_store 重建完整上下文。

        Args:
            relevant_items: fact_id 列表（如 ["fact_0", "fact_2"]），
                          如果为 None 则使用所有事实

        Returns:
            重建的完整上下文字符串
        """
        if not self.knowledge_base:
            return ""

        # 确定要使用哪些事实
        if relevant_items is None:
            # 使用所有事实
            selected_facts = self.knowledge_base
        else:
            # 根据 fact_id 筛选事实
            selected_facts = []
            for fact_id in relevant_items:
                # 解析 fact_id (格式: "fact_0", "fact_1", ...)
                if fact_id.startswith("fact_"):
                    try:
                        index = int(fact_id.split("_")[1])
                        if 0 <= index < len(self.knowledge_base):
                            selected_facts.append(self.knowledge_base[index])
                    except (ValueError, IndexError):
                        continue

        # 收集所有相关的 chunk_ids
        all_chunk_ids = []
        for fact in selected_facts:
            all_chunk_ids.extend(fact.chunk_ids)

        # 去重并保持顺序
        unique_chunk_ids = []
        seen = set()
        for chunk_id in all_chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_chunk_ids.append(chunk_id)

        # 从 chunk_store 提取原文
        chunks = []
        for chunk_id in unique_chunk_ids:
            if chunk_id in self.chunk_store:
                chunks.append(self.chunk_store[chunk_id])

        return "\n\n---\n\n".join(chunks)

    def _extract_facts(
        self,
        query: str,
        chunk_pairs: List[Tuple[str, str]],
    ) -> List[FactObject]:
        """从 chunks 中提取与 query 相关的事实。

        Args:
            query: 查询文本
            chunk_pairs: (chunk_id, chunk_text) 元组列表

        Returns:
            提取的 FactObject 列表
        """
        if not chunk_pairs:
            return []

        # 读取 prompt 模板
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "fact_extract.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # 构造 chunks JSON（使用索引而非 chunk_id）
        chunks_data = [
            {"index": i, "text": text} for i, (chunk_id, text) in enumerate(chunk_pairs)
        ]
        chunks_json = json.dumps(chunks_data, ensure_ascii=False, indent=2)

        # 填充 prompt
        prompt = prompt_template.format(query=query, chunks=chunks_json)

        breakpoint()

        # 调用 LLM
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(
            messages=messages,
            response_format={"type": "json_object"},
        )

        # 解析响应
        try:
            # LLM 可能返回 {"facts": [...]} 或直接返回 [...]
            result = json.loads(response)
            if isinstance(result, dict) and "facts" in result:
                facts_list = result["facts"]
            elif isinstance(result, list):
                facts_list = result
            else:
                facts_list = []
        except json.JSONDecodeError:
            facts_list = []

        # 将索引映射回 chunk_ids 和 URLs，构造 FactObject
        fact_objects = []
        for fact_data in facts_list:
            summary = fact_data.get("summary", "")
            chunk_indices = fact_data.get("chunk_indices", [])

            if not summary or not chunk_indices:
                continue

            # 映射回 chunk_ids
            chunk_ids = [
                chunk_pairs[i][0] for i in chunk_indices if i < len(chunk_pairs)
            ]

            if not chunk_ids:
                continue

            # 从第一个 chunk_id 获取 source_url
            first_chunk_id = chunk_ids[0]
            source_url = self.chunk_url_map.get(first_chunk_id, "")

            fact_objects.append(
                FactObject(
                    summary=summary,
                    chunk_ids=chunk_ids,
                    source_url=source_url,
                )
            )

        return fact_objects


__all__ = ["FactCentricStrategy"]
