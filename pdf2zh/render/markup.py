from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Display math: <math display="block">X</math>
_MATH_DISPLAY = re.compile(
    r'<math\s+display=["\']block["\'][^>]*>(.*?)</math>', re.DOTALL | re.IGNORECASE
)
# Inline math: <math>X</math>  (no display attribute, or display="inline")
_MATH_INLINE = re.compile(
    r'<math(?:\s+display=["\']inline["\'])?[^>]*>(.*?)</math>', re.DOTALL | re.IGNORECASE
)
_BOLD = re.compile(r'<(?:b|strong)>(.*?)</(?:b|strong)>', re.DOTALL | re.IGNORECASE)
_ITALIC = re.compile(r'<(?:i|em)>(.*?)</(?:i|em)>', re.DOTALL | re.IGNORECASE)
_SUP = re.compile(r'<sup>(.*?)</sup>', re.DOTALL | re.IGNORECASE)
_SUB = re.compile(r'<sub>(.*?)</sub>', re.DOTALL | re.IGNORECASE)
_ANY_TAG = re.compile(r'<[^>]+>')

# Bare LaTeX command sequences (outside $ markers): \cmd{...} or \cmd
_BARE_LATEX = re.compile(
    r'(?<!\$)'                          # not already inside $
    r'((?:\\[a-zA-Z]+(?:\{[^}]*\}|\[[^\]]*\])*\s*)+)'  # one or more \cmd{...}
)

# Typst special chars in plain text (outside math)
_TYPST_ESCAPE = re.compile(r'([#@\\])')
# Literal < > after all tags stripped
_LT = re.compile(r'<(?![a-zA-Z/])')
_GT = re.compile(r'(?<![a-zA-Z0-9"])\s*>')


def escape_typst_string(text: str) -> str:
    """Escape for embedding inside a Typst double-quoted string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def to_typst_markup(text: str, *, is_equation: bool = False) -> str:
    """Convert hybrid HTML/LaTeX text to cmarker-friendly markdown with mitex math.

    Args:
        text: The translated_text field value (may contain HTML tags and LaTeX).
        is_equation: True for EQUATION category elements — bare LaTeX gets wrapped.

    Returns:
        String safe for cmarker.render(..., math: mitex) in Typst.
    """
    if not text:
        return ""

    # 1. Display math → $$ ... $$ (double-dollar block math for mitex)
    result = _MATH_DISPLAY.sub(lambda m: f"$${m.group(1).strip()}$$", text)

    # 2. Inline math → $ ... $
    result = _MATH_INLINE.sub(lambda m: f"${m.group(1).strip()}$", result)

    # 3. Bold / italic
    result = _BOLD.sub(lambda m: f"**{m.group(1)}**", result)
    result = _ITALIC.sub(lambda m: f"_{m.group(1)}_", result)

    # 4. Superscript / subscript
    #    Inside existing $...$ context: leave as LaTeX (^{X}, _{X} handled by mitex)
    #    Outside math: use markdown superscript ^X^ / subscript ~X~
    result = _convert_sup_sub(result)

    # 5. For EQUATION elements, wrap bare LaTeX command runs in $...$
    if is_equation:
        result = _wrap_bare_latex(result)

    # 6. Strip any remaining unknown HTML tags
    result = _ANY_TAG.sub("", result)

    # 7. Escape literal < > that remain (comparison operators etc.)
    result = result.replace("\\<", "\x00LT\x00").replace("\\>", "\x00GT\x00")
    result = re.sub(r"<", r"\\<", result)
    result = re.sub(r">", r"\\>", result)
    result = result.replace("\x00LT\x00", "\\<").replace("\x00GT\x00", "\\>")

    # 8. Escape Typst-special chars in plain text segments (outside $...$)
    result = _escape_typst_outside_math(result)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_sup_sub(text: str) -> str:
    """Replace <sup>X</sup> and <sub>X</sub> preserving $ contexts."""
    parts = _split_math(text)
    out = []
    for kind, chunk in parts:
        if kind == "math":
            out.append(chunk)
        else:
            chunk = _SUP.sub(lambda m: f"^{m.group(1)}^", chunk)
            chunk = _SUB.sub(lambda m: f"~{m.group(1)}~", chunk)
            out.append(chunk)
    return "".join(out)


def _wrap_bare_latex(text: str) -> str:
    """Wrap bare LaTeX command sequences in $...$ (for EQUATION elements)."""
    parts = _split_math(text)
    out = []
    for kind, chunk in parts:
        if kind == "math":
            out.append(chunk)
        else:
            # Wrap runs of LaTeX commands that are not yet in math
            out.append(_BARE_LATEX.sub(lambda m: f"${m.group(1).strip()}$", chunk))
    return "".join(out)


def _escape_typst_outside_math(text: str) -> str:
    """Escape # and @ outside math delimiters (backslash already handled)."""
    parts = _split_math(text)
    out = []
    for kind, chunk in parts:
        if kind == "math":
            out.append(chunk)
        else:
            # Escape # and @ which have special meaning in Typst
            chunk = chunk.replace("#", "\\#").replace("@", "\\@")
            out.append(chunk)
    return "".join(out)


def _split_math(text: str) -> list[tuple[str, str]]:
    """Split text into alternating (kind, chunk) where kind is 'text' or 'math'.

    Handles $...$ and $$...$$ delimiters.
    """
    result: list[tuple[str, str]] = []
    i = 0
    n = len(text)
    buf = []

    while i < n:
        if text[i] == "$":
            # Flush text buffer
            if buf:
                result.append(("text", "".join(buf)))
                buf = []
            # Determine if $$ or $
            if i + 1 < n and text[i + 1] == "$":
                delim = "$$"
                i += 2
            else:
                delim = "$"
                i += 1
            # Find closing delimiter
            end = text.find(delim, i)
            if end == -1:
                # No closing delimiter — treat rest as text
                result.append(("text", delim + text[i:]))
                break
            math_content = text[i:end]
            result.append(("math", f"{delim}{math_content}{delim}"))
            i = end + len(delim)
        else:
            buf.append(text[i])
            i += 1

    if buf:
        result.append(("text", "".join(buf)))

    return result


# ---------------------------------------------------------------------------
# TOC line parser
# ---------------------------------------------------------------------------


def parse_toc_line(line: str) -> tuple[str, str] | None:
    """Extract (title, page_number) from a TOC entry line, or None if not a TOC line."""
    line = line.strip()
    if not line:
        return None
    m = re.match(r'^(.*?)\s+(\d+)\s*$', line)
    if m:
        return m.group(1).strip(), m.group(2)
    return None
