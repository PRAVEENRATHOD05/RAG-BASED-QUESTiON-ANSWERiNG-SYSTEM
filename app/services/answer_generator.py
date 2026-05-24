from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol

from app.models.entities import SearchHit

logger = logging.getLogger(__name__)

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[a-zA-Z0-9]{3,}")
STOPWORDS = {
    "what",
    "when",
    "where",
    "which",
    "about",
    "there",
    "their",
    "would",
    "could",
    "should",
    "from",
    "into",
    "that",
    "this",
    "with",
    "have",
    "your",
}


class AnswerGenerator(Protocol):
    def generate(self, question: str, hits: list[SearchHit]) -> str: ...


class ExtractiveAnswerGenerator:
    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return (
                "I could not find a confident answer in the indexed documents. "
                "Please ingest relevant files or rephrase the question."
            )

        keywords = self._keywords(question)
        candidates: list[tuple[float, str]] = []

        for hit in hits:
            sentences = SENTENCE_SPLIT_PATTERN.split(hit.chunk.text)
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if len(clean_sentence) < 35:
                    continue
                sentence_keywords = self._keywords(clean_sentence)
                overlap = len(keywords & sentence_keywords)
                score = (hit.score * 1.1) + (overlap * 0.25)
                candidates.append((score, clean_sentence))

        if not candidates:
            snippet = hits[0].chunk.text[:380].strip()
            return snippet if snippet else "No readable answer segment found."

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = self._select_unique_sentences(candidates)
        evidence_sources = ", ".join(
            sorted({Path(hit.chunk.source).name for hit in hits[:3]})
        )
        summary = " ".join(selected)
        return f"{summary}\n\nEvidence source(s): {evidence_sources}."

    def _keywords(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in WORD_PATTERN.findall(text)
            if token.lower() not in STOPWORDS
        }

    def _select_unique_sentences(self, candidates: list[tuple[float, str]]) -> list[str]:
        selected: list[str] = []
        seen_signatures: set[str] = set()

        for _, sentence in candidates:
            signature = sentence.lower()[:90]
            if signature in seen_signatures:
                continue
            selected.append(sentence)
            seen_signatures.add(signature)
            if len(selected) >= self.max_sentences:
                break

        return selected


class OpenAIAnswerGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI SDK is not installed. Install 'openai' package first."
            ) from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fallback = ExtractiveAnswerGenerator()

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return self.fallback.generate(question, hits)

        context = "\n\n".join(
            [
                f"[Source: {hit.chunk.source} | Score: {hit.score:.4f}]\n{hit.chunk.text}"
                for hit in hits
            ]
        )
        system_prompt = (
            "You are a careful assistant for document question answering. "
            "Use only the provided context. "
            "If context is insufficient, say that clearly."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer clearly and concisely. Do not invent facts not present in context."
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                temperature=0.15,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = (response.output_text or "").strip()
            return text if text else self.fallback.generate(question, hits)
        except Exception:  # noqa: BLE001
            logger.exception("OpenAI answer generation failed, using local fallback.")
            return self.fallback.generate(question, hits)


def build_answer_generator(
    use_openai: bool,
    openai_api_key: str | None,
    openai_model: str,
) -> AnswerGenerator:
    if use_openai and openai_api_key:
        try:
            return OpenAIAnswerGenerator(api_key=openai_api_key, model=openai_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OpenAI generator could not be initialized (%s). Falling back to local mode.",
                exc,
            )
    return ExtractiveAnswerGenerator()

