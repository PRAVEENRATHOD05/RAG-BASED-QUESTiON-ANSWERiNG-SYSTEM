from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol

from app.models.entities import SearchHit
from app.services.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+-]{1,}")
SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|[;\n]+")

INTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "skills": re.compile(r"\b(skill|tools?|stack|technology|languages?)\b", re.IGNORECASE),
    "projects": re.compile(r"\b(project|portfolio|build|developed)\b", re.IGNORECASE),
    "experience": re.compile(r"\b(experience|internship|intern|role|work)\b", re.IGNORECASE),
    "education": re.compile(r"\b(education|degree|college|university|course)\b", re.IGNORECASE),
    "achievements": re.compile(r"\b(achievement|award|rank|accomplishment)\b", re.IGNORECASE),
    "job_fit": re.compile(r"\b(job fit|fit for|qualified|match|hire)\b", re.IGNORECASE),
    "summary": re.compile(r"\b(summary|profile|about me|introduce)\b", re.IGNORECASE),
}

INTENT_SECTIONS: dict[str, tuple[str, ...]] = {
    "skills": ("skills",),
    "projects": ("projects",),
    "experience": ("experience", "internship"),
    "education": ("education",),
    "achievements": ("achievement", "award"),
    "summary": ("summary", "profile", "experience", "projects"),
    "job_fit": ("summary", "experience", "projects", "skills"),
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "about",
    "from",
    "into",
    "what",
    "when",
    "where",
    "which",
    "tell",
    "please",
    "resume",
    "profile",
    "me",
    "you",
    "your",
    "my",
}


class AnswerGenerator(Protocol):
    def generate(self, question: str, hits: list[SearchHit]) -> str: ...


class ExtractiveAnswerGenerator:
    def __init__(self, max_bullets: int = 5) -> None:
        self.max_bullets = max_bullets
        self.cleaner = TextCleaner()

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return self._not_found_response()

        intent = self._infer_intent(question)
        question_tokens = self._tokens(question)
        evidence = self._collect_evidence(question_tokens=question_tokens, hits=hits, intent=intent)
        if not evidence:
            return self._not_found_response()

        if self._is_insufficient(question_tokens=question_tokens, evidence=evidence, hits=hits):
            return self._not_found_response()

        bullets = self._build_bullets(intent=intent, evidence=evidence)
        confidence = self._confidence_level(hits=hits, question_tokens=question_tokens, evidence=evidence)
        sources = self._format_sources(evidence)
        return self._render_response(answer_lines=bullets, confidence=confidence, sources=sources)

    def _infer_intent(self, question: str) -> str:
        for intent, pattern in INTENT_PATTERNS.items():
            if pattern.search(question):
                return intent
        return "general"

    def _tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
            if token.lower() not in STOPWORDS
        }

    def _collect_evidence(
        self,
        question_tokens: set[str],
        hits: list[SearchHit],
        intent: str,
    ) -> list[tuple[float, str, str, str, int | None]]:
        collected: list[tuple[float, str, str, str, int | None]] = []
        section_preferences = INTENT_SECTIONS.get(intent, ())

        for hit in hits:
            section_name = str(hit.chunk.metadata.get("section_name", "General"))
            section_lower = section_name.lower()
            source_file = str(hit.chunk.metadata.get("source_filename") or Path(hit.chunk.source).name)
            page_number_raw = hit.chunk.metadata.get("page_number")
            page_number = int(page_number_raw) if isinstance(page_number_raw, int) else None

            snippets = [snippet.strip() for snippet in SPLIT_PATTERN.split(hit.chunk.text) if snippet.strip()]
            for snippet in snippets:
                cleaned = self.cleaner.normalize_excerpt(snippet, max_chars=280)
                if len(cleaned) < 38:
                    continue

                snippet_tokens = self._tokens(cleaned)
                overlap = len(question_tokens & snippet_tokens)
                overlap_ratio = overlap / max(len(question_tokens), 1)

                score = (hit.score * 1.2) + (overlap_ratio * 0.6)
                if section_preferences and any(name in section_lower for name in section_preferences):
                    score += 0.18
                if cleaned.startswith("•") or cleaned.startswith("-"):
                    score += 0.05
                if re.search(r"\b\d+[%+x]?\b", cleaned):
                    score += 0.07

                collected.append((score, cleaned, source_file, section_name, page_number))

        ranked = sorted(collected, key=lambda item: item[0], reverse=True)
        if section_preferences:
            preferred_ranked = [
                item
                for item in ranked
                if any(name in item[3].lower() for name in section_preferences)
            ]
            if preferred_ranked:
                ranked = preferred_ranked + [
                    item
                    for item in ranked
                    if item not in preferred_ranked
                ]

        deduped: list[tuple[float, str, str, str, int | None]] = []
        seen: set[str] = set()

        for item in ranked:
            signature = item[1].lower()[:150]
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(item)
        return deduped

    def _is_insufficient(
        self,
        question_tokens: set[str],
        evidence: list[tuple[float, str, str, str, int | None]],
        hits: list[SearchHit],
    ) -> bool:
        if not evidence or not hits:
            return True

        if len(question_tokens) < 2:
            return False

        best_overlap_ratio = 0.0
        for _, snippet, _, _, _ in evidence[:4]:
            snippet_tokens = self._tokens(snippet)
            overlap = len(question_tokens & snippet_tokens)
            best_overlap_ratio = max(best_overlap_ratio, overlap / len(question_tokens))

        top_hit_score = hits[0].score
        return best_overlap_ratio < 0.18 and top_hit_score < 0.42

    def _build_bullets(
        self,
        intent: str,
        evidence: list[tuple[float, str, str, str, int | None]],
    ) -> list[str]:
        if intent == "summary":
            return self._summary_bullets(evidence)
        if intent == "job_fit":
            return self._job_fit_bullets(evidence)

        bullets: list[str] = []
        for _, snippet, _, _, _ in evidence[: self.max_bullets]:
            bullets.append(f"- {snippet.lstrip('-• ').strip()}")
        return bullets

    def _summary_bullets(self, evidence: list[tuple[float, str, str, str, int | None]]) -> list[str]:
        lines = [
            "- ATS Summary: Candidate demonstrates hands-on project and/or internship evidence in the retrieved resume context."
        ]
        for _, snippet, _, _, _ in evidence[:4]:
            lines.append(f"- {snippet.lstrip('-• ').strip()}")
        return lines

    def _job_fit_bullets(self, evidence: list[tuple[float, str, str, str, int | None]]) -> list[str]:
        lines = [
            "- Job-fit analysis is based only on retrieved resume evidence."
        ]
        for _, snippet, _, _, _ in evidence[:4]:
            lines.append(f"- {snippet.lstrip('-• ').strip()}")
        return lines

    def _confidence_level(
        self,
        hits: list[SearchHit],
        question_tokens: set[str],
        evidence: list[tuple[float, str, str, str, int | None]],
    ) -> str:
        if not hits or not evidence:
            return "Low"

        top_score = hits[0].score
        avg_top = sum(hit.score for hit in hits[:3]) / min(3, len(hits))

        best_overlap_ratio = 0.0
        for _, snippet, _, _, _ in evidence[:3]:
            overlap = len(question_tokens & self._tokens(snippet))
            if question_tokens:
                best_overlap_ratio = max(best_overlap_ratio, overlap / len(question_tokens))

        if top_score >= 0.72 and avg_top >= 0.58 and best_overlap_ratio >= 0.45:
            return "High"
        if top_score >= 0.48 and avg_top >= 0.32 and best_overlap_ratio >= 0.2:
            return "Medium"
        return "Low"

    def _format_sources(self, evidence: list[tuple[float, str, str, str, int | None]]) -> list[str]:
        source_lines: list[str] = []
        seen: set[str] = set()

        for _, _, filename, section_name, page_number in evidence:
            page_part = f"page {page_number}" if page_number else "page n/a"
            entry = f"{filename} - {section_name} ({page_part})"
            if entry in seen:
                continue
            seen.add(entry)
            source_lines.append(entry)
            if len(source_lines) >= 4:
                break
        return source_lines

    def _render_response(self, answer_lines: list[str], confidence: str, sources: list[str]) -> str:
        if not answer_lines:
            answer_lines = ["- I could not find that information in the uploaded document."]
        if not sources:
            sources = ["Not available"]

        answer_block = "\n".join(answer_lines)
        sources_block = "\n".join(f"- {source}" for source in sources)
        return (
            "Answer:\n"
            f"{answer_block}\n\n"
            "Confidence:\n"
            f"{confidence}\n\n"
            "Sources:\n"
            f"{sources_block}"
        )

    def _not_found_response(self) -> str:
        return (
            "Answer:\n"
            "I could not find that information in the uploaded document.\n\n"
            "Confidence:\n"
            "Low\n\n"
            "Sources:\n"
            "- Not available"
        )


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
                (
                    f"[File: {Path(hit.chunk.source).name} | "
                    f"Section: {hit.chunk.metadata.get('section_name', 'General')} | "
                    f"Page: {hit.chunk.metadata.get('page_number', 'n/a')} | "
                    f"Score: {hit.score:.4f}]\n{hit.chunk.text}"
                )
                for hit in hits
            ]
        )
        system_prompt = (
            "You are an enterprise RAG Resume Assistant for recruiters.\n"
            "Rules:\n"
            "- Use only provided context.\n"
            "- Do not fabricate.\n"
            "- If unavailable, return: I could not find that information in the uploaded document.\n"
            "- Keep concise and recruiter-friendly.\n"
            "- Use bullets for skills/projects/tools/experience.\n"
            "- Highlight measurable impact when present.\n"
            "- For summaries, produce ATS-friendly wording.\n"
            "- For job-fit, compare only with provided resume data.\n\n"
            "Output format exactly:\n"
            "Answer:\n<response>\n\n"
            "Confidence:\n<High / Medium / Low>\n\n"
            "Sources:\n- filename - section/page"
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            "Follow the output format exactly."
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                temperature=0.05,
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
