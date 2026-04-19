from __future__ import annotations

import json


def build_translation_prompt(
    chunk: dict[str, str],
    src_lang: str,
    tgt_lang: str,
    glossary_block: str,
) -> tuple[str, str]:
    system = (
        f"You are a professional, authentic machine translation engine.\n\n"
        f"# Task\nTranslate text from {src_lang} into {tgt_lang}.\n\n"
        f"# Rules\n"
        f"1. Preserve verbatim: content inside <math>...</math> or $...$, URLs, code blocks, "
        f"brand names, and any <ph-xxx> placeholder tags.\n"
        f"2. LENGTH CONSTRAINT (hard): each translation's character length MUST be within "
        f"±15% of its source string. Rephrase compactly if needed.\n"
        f"   Priority order when in tension:\n"
        f"     (1) semantic accuracy\n"
        f"     (2) length preservation\n"
        f"3. Do NOT merge or split entries. Every input id must appear exactly once in the "
        f"output, with the same id string.\n"
        f"4. Return ONLY the JSON array specified below. No prose, no code fences.\n\n"
        f"{glossary_block}"
    )
    input_json = json.dumps(chunk, ensure_ascii=False)
    example_ids = list(chunk.keys())[:2]
    example = ", ".join(f'{{"id":"{k}","t":"<translation {k}>"}}' for k in example_ids)
    user = (
        f"<input>\n```json\n{input_json}\n```\n</input>\n\n"
        f"Return a JSON array in exactly this shape:\n[{example}]"
    )
    return system, user


def build_glossary_prompt(
    chunk: dict[str, str],
    src_lang: str,
    tgt_lang: str,
) -> tuple[str, str]:
    system = "You are a professional glossary extractor."
    input_json = json.dumps(chunk, ensure_ascii=False)
    user = (
        f"Extract proper nouns — people, places, organizations, product names, technical terms "
        f"— from the {src_lang} text below. Provide their {tgt_lang} translations.\n\n"
        f"Rules:\n"
        f"- Do NOT include common nouns.\n"
        f"- Do NOT include content inside <math>...</math> or <ph-xxx> tags.\n"
        f"- Each src appears at most once. No explanations.\n\n"
        f"<input>\n```json\n{input_json}\n```\n</input>\n\n"
        f'Output format — JSON array only:\n[{{"src":"<term>","dst":"<translation>"}}]'
    )
    return system, user


def glossary_block_for_chunk(chunk: dict[str, str], glossary: dict[str, str]) -> str:
    combined = " ".join(chunk.values()).lower()
    matches = [(src, dst) for src, dst in glossary.items() if src.lower() in combined]
    if not matches:
        return ""
    lines = "\n".join(f"{src} => {dst}" for src, dst in matches)
    return f"# Glossary (use these exact translations when the term appears)\n{lines}\n"
