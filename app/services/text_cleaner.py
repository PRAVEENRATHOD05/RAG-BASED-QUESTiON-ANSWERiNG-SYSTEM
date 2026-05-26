from __future__ import annotations

import re
import unicodedata

COMMON_SECTION_NAMES = {
    "summary",
    "profile",
    "objective",
    "experience",
    "work experience",
    "internship",
    "internships",
    "projects",
    "education",
    "skills",
    "technical skills",
    "achievements",
    "certifications",
    "contact",
}


class TextCleaner:
    """Normalize noisy OCR/PDF text into cleaner retrieval-ready text."""

    _ligature_map = str.maketrans(
        {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "–": "-",
            "—": "-",
            "‑": "-",
            "•": " • ",
            "●": " • ",
            "▪": " • ",
            "·": " • ",
            "“": '"',
            "”": '"',
            "’": "'",
            "‘": "'",
        }
    )

    _control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
    _multi_space = re.compile(r"[ \t]{2,}")
    _multi_newline = re.compile(r"\n{3,}")
    _space_before_punct = re.compile(r"\s+([,.;:!?])")
    _missing_space_after_punct = re.compile(r"([,.;:!?])([A-Za-z0-9])")
    _letter_digit = re.compile(r"([A-Za-z])([0-9])")
    _digit_letter = re.compile(r"([0-9])([A-Za-z])")
    _acronym_boundary = re.compile(r"([A-Z]{2,})([A-Z][a-z])")
    _camel_case = re.compile(r"([a-z])([A-Z])")
    _broken_pipe = re.compile(r"\|(?=\S)")
    _ocr_symbol_noise = re.compile(r"[⌢⌣⌡⌠◦◌]+")
    _url_noise = re.compile(r"(?<=\w)/(?!\s)")
    _hyphenated_line_break = re.compile(r"([A-Za-z])-\n([A-Za-z])")
    _line_with_no_spacing = re.compile(r"\b([A-Za-z]{18,})\b")
    _word_glue_markers = (
        "experience",
        "education",
        "projects",
        "skills",
        "summary",
        "technology",
        "internship",
        "manager",
        "engineer",
        "developer",
        "analysis",
        "retrieval",
        "generation",
    )

    def clean(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text or "")
        normalized = normalized.translate(self._ligature_map)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = self._control_chars.sub(" ", normalized)
        normalized = self._hyphenated_line_break.sub(r"\1\2", normalized)
        normalized = self._ocr_symbol_noise.sub(" ", normalized)
        normalized = self._url_noise.sub("/ ", normalized)
        normalized = self._broken_pipe.sub("| ", normalized)
        normalized = self._missing_space_after_punct.sub(r"\1 \2", normalized)
        normalized = self._space_before_punct.sub(r"\1", normalized)
        normalized = self._letter_digit.sub(r"\1 \2", normalized)
        normalized = self._digit_letter.sub(r"\1 \2", normalized)
        normalized = self._acronym_boundary.sub(r"\1 \2", normalized)
        normalized = self._camel_case.sub(r"\1 \2", normalized)
        normalized = self._repair_glued_words(normalized)
        normalized = self._multi_space.sub(" ", normalized)
        normalized = re.sub(r"[ \t]+\n", "\n", normalized)
        normalized = re.sub(r"\n[ \t]+", "\n", normalized)
        normalized = self._multi_newline.sub("\n\n", normalized)
        return normalized.strip()

    def normalize_excerpt(self, text: str, max_chars: int = 260) -> str:
        compact = self.clean(text).replace("\n", " ")
        compact = self._multi_space.sub(" ", compact).strip()
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    def infer_section_name(self, text: str, fallback: str = "General") -> str:
        lines = [line.strip(" :|-").strip() for line in text.splitlines() if line.strip()]
        for line in lines[:8]:
            lowered = line.lower()
            if lowered in COMMON_SECTION_NAMES:
                return line.title()
            if len(line) <= 40 and any(name in lowered for name in COMMON_SECTION_NAMES):
                return line.title()
        return fallback

    def _repair_glued_words(self, text: str) -> str:
        def repair_token(match: re.Match[str]) -> str:
            token = match.group(1)
            lowered = token.lower()

            for marker in self._word_glue_markers:
                marker_index = lowered.find(marker)
                if marker_index > 2:
                    head = token[:marker_index]
                    tail = token[marker_index:]
                    if head.isalpha() and tail.isalpha():
                        return f"{head} {tail}"
            return token

        return self._line_with_no_spacing.sub(repair_token, text)
