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
        f"1. For content inside <math>...</math> tags: convert LaTeX to Typst math syntax "
        f"(do NOT use backslash commands). Conversion table:\n"
        f"   \\frac{{a}}{{b}} → frac(a, b)  |  \\sqrt{{x}} → sqrt(x)  |  \\pi → pi  |  \\theta → theta\n"
        f"   \\alpha → alpha  |  \\beta → beta  |  \\gamma → gamma  |  \\delta → delta\n"
        f"   \\sigma → sigma  |  \\mu → mu  |  \\lambda → lambda  |  \\omega → omega\n"
        f"   \\sin → sin  |  \\cos → cos  |  \\tan → tan  |  \\log → log  |  \\ln → ln\n"
        f"   \\sum_{{i}}^{{n}} → sum_(i)^(n)  |  \\int_{{a}}^{{b}} → integral_a^b\n"
        f"   \\binom{{n}}{{k}} → binom(n, k)  |  \\pm → plus.minus  |  \\times → times\n"
        f"   \\leq → <=  |  \\geq → >=  |  \\neq → !=  |  \\infty → oo\n"
        f"   \\left( ... \\right) → ( ... )  (just drop \\left/\\right)\n"
        f"   \\begin{{pmatrix}} a & b \\\\\\\\ c & d \\end{{pmatrix}} → mat(a, b; c, d)\n"
        f"   \\cdots → dots.c  |  \\ldots → dots.b  |  \\vdots → dots.v  |  \\ddots → dots.down\n"
        f'   ^{{xy}} → ^(xy)  |  _{{xy}} → _(xy)  |  \\text{{word}} → "word"\n'
        f"   Keep <math>...</math> tags around the converted content.\n"
        f"   IMPORTANT: In Typst math, adjacent letters like `bh` form ONE identifier.\n"
        f"   Always separate single-letter variables with spaces: `b h` not `bh`, `a b` not `ab`.\n"
        f"   Example: <math>\\frac{{1}}{{2}}bh</math> → <math>frac(1, 2) b h</math>\n"
        f"   EXCEPTION — single-symbol inline mentions in prose: when <math>X</math> wraps a\n"
        f"   bare single letter/symbol used as a variable reference inside running prose\n"
        f"   (NOT part of a multi-term formula), UNWRAP it and render the symbol as italic\n"
        f"   text with <i>...</i> instead. This keeps the sentence flowing on one line.\n"
        f"   Example: 'area <math>A</math>, circumference <math>C</math>' →\n"
        f"            'diện tích <i>A</i>, chu vi <i>C</i>'\n"
        f"   Keep <math>...</math> for any expression with ≥2 tokens or an operator\n"
        f"   (e.g. <math>x = y</math>, <math>r^2</math>, <math>\\pi r</math> all stay wrapped).\n"
        f"2. Preserve verbatim: content inside $...$, HTML tags <b>, <i>, <sup>, <sub>, "
        f"URLs, code blocks, brand names, and any <ph-xxx> placeholder tags.\n"
        f"3. LENGTH CONSTRAINT (hard): each translation's character length MUST be within "
        f"±15% of its source string. Rephrase compactly if needed.\n"
        f"   Priority order when in tension:\n"
        f"     (1) semantic accuracy\n"
        f"     (2) length preservation\n"
        f"4. Do NOT merge or split entries. Every input id must appear exactly once in the "
        f"output, with the same id string.\n"
        f"5. If an entry contains ONLY mathematical notation — Greek letter names "
        f"(pi, theta, alpha…), function names (sin, cos, log…), unit symbols "
        f"(rad, deg, rpm…), or operator words (plus, minus…) with no natural-language "
        f"prose — return it VERBATIM unchanged. Do not translate or paraphrase.\n"
        f"6. Return ONLY the JSON array specified below. No prose, no code fences.\n\n"
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


def build_toc_fix_prompt(
    entries: list[dict],
) -> tuple[str, str]:
    """Prompt for the post-translation TOC-fix pass.

    Each entry: {"id": str, "text": str, "bbox": [x0, y0, x1, y1]}
    LLM returns: [{"id": str, "t": str}]
    """
    system = (
        "You are a Table-of-Contents reconstructor.\n\n"
        "# Task\n"
        "Each input entry is a TOC concatenated into one line — section numbers, titles,\n"
        "dot leaders, and page numbers all run together. Restructure it so each entry sits\n"
        "on its own line, formatted as: '<title>\\t<page_number>'.\n\n"
        "## Rules\n"
        "1. ONE entry per line. Separator between title and page number is a single tab '\\t'.\n"
        "2. Preserve the section number prefix in the title (e.g. '1.1 Bối cảnh').\n"
        "3. Strip dot-leader sequences ('. . . . .') — they are layout fillers.\n"
        "4. Keep the page number as the integer that ends the entry (1–3 digits typically).\n"
        "5. If an entry has no page number visible, output just the title (no tab).\n"
        "6. Do NOT translate or rewrite titles. Preserve them as-is.\n"
        "7. Do NOT add extra blank lines or commentary.\n\n"
        "## Example\n"
        "Input:  '1 Giới thiệu 6 1.1 Bối cảnh về LLM . . . . . . . . . 6 1.2 Lịch sử . . . . . . 7'\n"
        "Output: '1 Giới thiệu\\t6\\n1.1 Bối cảnh về LLM\\t6\\n1.2 Lịch sử\\t7'\n\n"
        "## Output format\n"
        "Return ONLY a JSON array. No prose, no fences. Every input id appears once.\n"
    )
    input_json = json.dumps(entries, ensure_ascii=False)
    example_ids = [str(e.get("id")) for e in entries[:2]]
    example = ", ".join(
        f'{{"id":"{k}","t":"<entry1>\\\\t<page>\\\\n<entry2>\\\\t<page>"}}'
        for k in example_ids
    )
    user = (
        f"<input>\n```json\n{input_json}\n```\n</input>\n\n"
        f"Return JSON array in this shape:\n[{example}]"
    )
    return system, user


def build_math_fix_prompt(
    entries: list[dict],
) -> tuple[str, str]:
    """Prompt for the post-translation math-fix pass.

    Each entry: {"id": str, "text": str, "bbox": [x0, y0, x1, y1]}
    LLM returns:  [{"id": str, "t": str}]
    """
    system = (
        "You are a Typst markup post-processor.\n\n"
        "# Task\n"
        "For each entry, FIX the translated text so it renders correctly in Typst:\n\n"
        "## Rule 1 — Wrap bare math in <math>...</math>\n"
        "Any math expression OUTSIDE existing <math> tags must be wrapped. Math means:\n"
        "- Contains Typst function call: frac(...), sqrt(...), sum(...), int(...)\n"
        "- Contains math keywords: pi, theta, alpha, sin, cos, tan, log, ln, infty\n"
        "- Has = or +/-/*/÷/× joining symbol-like terms (variables, numbers, math fns)\n"
        "- Has ^ or _ for exponent/subscript: x^2, a_n\n"
        "Example: 'A = pi r sqrt(r^2 + h^2)' → '<math>A = pi r sqrt(r^2 + h^2)</math>'\n"
        "Example: 'tan x = frac(sin x, cos x)' → '<math>tan x = frac(sin x, cos x)</math>'\n\n"
        "## Rule 2 — Convert LaTeX math to Typst math syntax\n"
        "Inside <math> tags, NO backslash commands. Use:\n"
        "  \\frac{a}{b} → frac(a, b)  |  \\sqrt{x} → sqrt(x)  |  \\pi → pi\n"
        "  \\sum_{i}^{n} → sum_(i)^(n)  |  \\int_a^b → integral_a^b\n"
        "  \\binom{n}{k} → binom(n, k)  |  \\pm → plus.minus  |  \\leq → <=\n"
        '  \\left( ... \\right) → ( ... )  |  \\text{w} → "w"\n\n'
        "## Rule 3 — Use <typst> blocks ONLY for grid layouts (be conservative)\n"
        "Default action: keep <math>...</math> tags + plain text as-is. DO NOT wrap simple\n"
        "text+math entries in <typst>. Only emit a <typst> block when the entry truly needs\n"
        "a multi-column / multi-row Typst grid (bbox is wide AND has multiple labels+formulas).\n\n"
        "Inside <typst>...</typst>: NO LaTeX backslash commands, NO escaped slashes.\n"
        "Use Typst syntax only: pi (not \\\\pi), frac(a,b) or a/b (not \\\\frac{}{}), sqrt(x) (not \\\\sqrt{}).\n\n"
        "Use bbox = [x0, y0, x1, y1] to compute width = x1-x0, height = y1-y0, aspect = width/height.\n\n"
        "**Pattern A — Label-then-Formula reference card** (most common):\n"
        "Entry has N short labels concatenated, then N math blocks. Always use 2 rows:\n"
        "  row 1 = labels, row 2 = corresponding formulas (one cell per pair).\n"
        "  Input:  'Tam giác Đường tròn Hình quạt tròn <math>A = frac(1,2) b h</math> <math>A = pi r^2</math> <math>A = frac(1,2) r^2 theta</math>'\n"
        "  Output: <typst>#grid(columns: 3, gutter: 8pt, [Tam giác], [Đường tròn], [Hình quạt tròn], [$A = b h / 2$], [$A = pi r^2$], [$A = r^2 theta / 2$])</typst>\n\n"
        "**Pattern B — One-label-one-or-more-formulas** (label with stacked formulas):\n"
        "Entry has 1 SHORT NOUN LABEL (shape name, object name) + 1 or more formulas.\n"
        "The label must be a noun/name — NOT a sentence fragment with conjunctions.\n"
        "Use a column grid (one cell per line):\n"
        "  Input:  'Hình cầu <math>V = frac(4,3) pi r^3</math> <math>A = 4 pi r^2</math>'\n"
        "  Output: <typst>#grid(columns: 1, row-gutter: 4pt, [Hình cầu], [$V = (4 pi r^3) / 3$], [$A = 4 pi r^2$])</typst>\n"
        "  Input:  'Hình trụ <math>V = pi r^2 h</math>'\n"
        "  Output: <typst>#grid(columns: 1, row-gutter: 4pt, [Hình trụ], [$V = pi r^2 h$])</typst>\n"
        "  DO NOT apply this pattern to prose sentences:\n"
        "  'If <math>a = b</math> then <math>c = d</math>' → keep as-is (prose with inline math, not a label)\n\n"
        "**Pattern C — Wide horizontal list** (≥3 math blocks, aspect>5, no labels):\n"
        "  <typst>#grid(columns: N, gutter: 8pt, [$...$], [$...$], ...)</typst>\n\n"
        "**Inside [content] cells — IMPORTANT formatting rules**:\n"
        "- Use $...$ for math (NOT <math> tags, since <typst> bypasses HTML processing).\n"
        "- NEVER use backslash inside content blocks — no \\\\, no \\/. Plain `/` for division.\n"
        "- Prefer 'a / b' over 'frac(a, b)' for inline fractions (better baseline alignment).\n"
        "  Example: $A = b h / 2$ — NOT $A = b h \\/ 2$, NOT $A = frac(1,2) b h$.\n"
        "- Reserve frac(...) only for stacked display fractions in tall bboxes.\n"
        "- Output one self-contained <typst>...</typst> per entry. Always close </typst>.\n\n"
        "## Rule 4 — Preserve non-math text exactly\n"
        "Do not translate, do not change wording. Only fix structure/wrapping.\n"
        "Preserve <i>X</i> single-letter italics as-is — do NOT re-wrap them in <math>.\n\n"
        "## Output\n"
        "Return ONLY a JSON array. No prose, no fences. Every input id must appear once.\n"
    )
    input_json = json.dumps(entries, ensure_ascii=False)
    example_ids = [str(e.get("id")) for e in entries[:2]]
    example = ", ".join(f'{{"id":"{k}","t":"<fixed text {k}>"}}' for k in example_ids)
    user = (
        f"<input>\n```json\n{input_json}\n```\n</input>\n\n"
        f"Return JSON array in this shape:\n[{example}]"
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
