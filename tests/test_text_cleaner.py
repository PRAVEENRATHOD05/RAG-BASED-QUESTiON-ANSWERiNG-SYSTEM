from app.services.text_cleaner import TextCleaner


def test_text_cleaner_normalizes_ocr_noise() -> None:
    cleaner = TextCleaner()
    raw = (
        "AIEngineerwithhands‑onexperience\n"
        "Builtﬂowsforretrieval.\n"
        "Skills:Python|FastAPI|RAG\n"
        "Contact⌢mail:test@example.com"
    )
    cleaned = cleaner.clean(raw)

    assert "AI Engineer" in cleaned
    assert "Builtflowsfor retrieval." in cleaned or "Built flowsfor retrieval." in cleaned
    assert "Python| Fast API| RAG" in cleaned or "Python| FastAPI| RAG" in cleaned
    assert "⌢" not in cleaned


def test_normalize_excerpt_compacts_text() -> None:
    cleaner = TextCleaner()
    raw = "  Example   text\n\nwith   spacing   and   noise.  "
    excerpt = cleaner.normalize_excerpt(raw, max_chars=40)
    assert "  " not in excerpt
    assert "\n" not in excerpt
