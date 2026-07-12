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
    r'<math(?:\s+display=["\']inline["\'])?[^>]*>(.*?)</math>',
    re.DOTALL | re.IGNORECASE,
)
_BOLD = re.compile(r"<(?:b|strong)>(.*?)</(?:b|strong)>", re.DOTALL | re.IGNORECASE)
_ITALIC = re.compile(r"<(?:i|em)>(.*?)</(?:i|em)>", re.DOTALL | re.IGNORECASE)
_SUP = re.compile(r"<sup>(.*?)</sup>", re.DOTALL | re.IGNORECASE)
_SUB = re.compile(r"<sub>(.*?)</sub>", re.DOTALL | re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")

# Bare LaTeX command sequences (outside $ markers): \cmd{...} or \cmd
_BARE_LATEX = re.compile(
    r"(?<!\$)"  # not already inside $
    r"((?:\\[a-zA-Z]+(?:\{[^}]*\}|\[[^\]]*\])*\s*)+)"  # one or more \cmd{...}
)

# Typst special chars in plain text (outside math)
_TYPST_ESCAPE = re.compile(r"([#@\\])")
# Literal < > after all tags stripped
_LT = re.compile(r"<(?![a-zA-Z/])")
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

    # If not in equation mode, escape literal dollar signs that are outside of <math> tags
    if not is_equation:
        _MATH_TAG = re.compile(r"<math\b[^>]*>.*?</math>", re.DOTALL | re.IGNORECASE)
        math_blocks = []

        def _stash_math(m: re.Match) -> str:
            math_blocks.append(m.group(0))
            return f"\x02MATH{len(math_blocks) - 1}\x03"

        result = _MATH_TAG.sub(_stash_math, text)
        result = result.replace("$", "\\$")
        result = re.sub(
            r"\x02MATH(\d+)\x03",
            lambda m: math_blocks[int(m.group(1))],
            result,
        )
    else:
        result = text

    # 1. Display math → $$ ... $$ (double-dollar block math for mitex)
    result = _MATH_DISPLAY.sub(lambda m: f"$${m.group(1).strip()}$$", result)

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

    # 6. Strip any remaining unknown HTML tags (preserving < > inside $...$)
    result = _strip_tags_outside_math(result)

    # 7 & 8. Escape Typst-special chars and literal < > in plain text segments (outside $...$)
    result = _escape_typst_outside_math(result, clean_math=False, escape_lt_gt=True)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_tags_outside_math(text: str) -> str:
    """Strip HTML-like tags but preserve $...$ math regions.

    Naive `_ANY_TAG.sub` would treat math relations like ``< b $ ... $ c >``
    as an HTML tag and erase the whole span; splitting on math first keeps
    the operators intact.
    """
    parts = _split_math(text)
    out = []
    for kind, chunk in parts:
        if kind == "math":
            out.append(chunk)
        else:
            out.append(_ANY_TAG.sub("", chunk))
    return "".join(out)


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


def _escape_typst_outside_math(text: str, *, clean_math: bool = False, escape_lt_gt: bool = False) -> str:
    """Escape # and @ outside math delimiters; clean up LaTeX inside math if clean_math is True."""
    parts = _split_math(text)
    out = []
    for kind, chunk in parts:
        if kind == "math":
            if clean_math:
                out.append(_clean_math_chunk(chunk))
            else:
                out.append(chunk)
        else:
            chunk = chunk.replace("#", "\\#").replace("@", "\\@")
            if escape_lt_gt:
                chunk = chunk.replace("\\<", "\x00LT\x00").replace("\\>", "\x00GT\x00")
                chunk = re.sub(r"<", r"\\<", chunk)
                chunk = re.sub(r">", r"\\>", chunk)
                chunk = chunk.replace("\x00LT\x00", "\\<").replace("\x00GT\x00", "\\>")
            out.append(chunk)
    return "".join(out)


def _clean_math_chunk(chunk: str) -> str:
    """Best-effort LaTeX → Typst conversion inside $...$ / $$...$$ regions.

    Handles common cases that the math-fixer LLM may have missed:
      \\frac{a}{b}  → frac(a, b)
      \\binom{a}{b} → binom(a, b)
      \\cmd{x}      → cmd(x)
      \\cmd         → cmd  (bare backslash command)
    Also runs identifier splitting (bh → b h).
    """
    m = re.match(r"^(\$+)(.*?)(\$+)$", chunk, re.DOTALL)
    if not m:
        return chunk
    open_d, content, close_d = m.group(1), m.group(2), m.group(3)
    # \limits and \nolimits are handled by Typst natively on operators, but raw \limits breaks Typst syntax. Strip them.
    content = re.sub(r"\\(?:no)?limits(?![a-zA-Z])", "", content)
    # \sqrt[n]{x} -> root(n, x)
    content = re.sub(r"\\sqrt\s*\[([^\[\]]+)\]\s*\{([^{}]*)\}", r"root(\1, \2)", content)
    # Two-arg LaTeX commands (frac/binom variants) — convert before single-arg pass
    content = re.sub(
        r"\\(?:frac|dfrac|tfrac|cfrac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        r"frac(\1, \2)",
        content,
    )
    content = re.sub(
        r"\\(?:binom|dbinom|tbinom)\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        r"binom(\1, \2)",
        content,
    )
    # Single-arg LaTeX command: \cmd{x} or \cmd*{x} → cmd(x)
    content = re.sub(r"\\([a-zA-Z]+)\*?\s*\{([^{}]*)\}", r"\1(\2)", content)
    # Bare backslash command: \cmd → cmd
    content = re.sub(r"\\([a-zA-Z]+)", r"\1", content)
    # Drop \left / \right artefacts (already stripped above as 'left'/'right')
    content = re.sub(r"\b(left|right)\s*([({[\])])", r"\2", content)
    content = _split_math_vars(content)
    return f"{open_d}{content}{close_d}"


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
# Typst native markup converter (for text with <math>Typst syntax</math>)
# ---------------------------------------------------------------------------

_MATH_TYPST_DISPLAY = re.compile(
    r'<math\s+display=["\']block["\'][^>]*>(.*?)</math>', re.DOTALL | re.IGNORECASE
)
_MATH_TYPST_INLINE = re.compile(
    r'<math(?:\s+display=["\']inline["\'])?[^>]*>(.*?)</math>',
    re.DOTALL | re.IGNORECASE,
)
# Raw Typst blocks emitted by the math-fix pass (e.g. #grid for layouts)
_TYPST_BLOCK = re.compile(r"<typst>(.*?)</typst>", re.DOTALL | re.IGNORECASE)

# Detect translatable prose: word run after stripping math/tags. Anything left
# means there's real text to render; otherwise the element is pure math and we
# should preserve the original PDF text layer.
_PROSE_LETTER_RUN = re.compile(
    r"[A-Za-zÀ-ɏḀ-ỿ"  # Latin + Latin Extended Additional (Vietnamese)
    r"Ͱ-ϿЀ-ӿ؀-ۿऀ-ॿ฀-๿⺀-鿿]{2,}"
)
_DOLLAR_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_DOLLAR_INLINE_RE = re.compile(r"\$[^$\n]*\$")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+(?:\s*\{[^{}]*\})*")
_EQ_LABEL_RE = re.compile(r"\(\d+(?:\.\d+)*[a-z]?\)")
_HTML_ANY_RE = re.compile(r"<[^>]+>")


def has_unbalanced_math_tags(text: str) -> bool:
    """Detect malformed <math>/<typst> with unmatched open/close counts.

    LLM outputs are sometimes truncated mid-stream; rendering such content as
    Typst markup produces 'unclosed delimiter' errors. Flag and skip them.
    """
    if not text:
        return False
    for tag in ("math", "typst"):
        opens = len(re.findall(rf"<{tag}\b", text, re.IGNORECASE))
        closes = len(re.findall(rf"</{tag}\b", text, re.IGNORECASE))
        if opens != closes:
            return True
    return False


# frac() with empty second arg: frac(x, ) — Typst math crash
_EMPTY_FRAC_RE = re.compile(r"frac\([^)]*,\s*\)")

# Letter-digit-letter inside math regions — unknown variable in Typst (e.g. t2c).
# Signals garbled LLM output (e.g. \cdot2\cdot converted incorrectly).
_MATH_REGION = re.compile(
    r"\$([^$]+)\$|<math\b[^>]*>(.*?)</math>", re.DOTALL | re.IGNORECASE
)
_ALPHA_DIGIT_ALPHA = re.compile(r"[a-zA-Z][0-9][a-zA-Z]")


def has_malformed_typst_math(text: str) -> bool:
    """True if text contains Typst math constructs that will cause a compile error.

    Detects:
    - frac() with empty denominator: frac(x, )
    - letter-digit-letter identifiers inside math regions: t2c (garbled LLM output)
    """
    if _EMPTY_FRAC_RE.search(text):
        return True
    for m in _MATH_REGION.finditer(text):
        content = m.group(1) or m.group(2) or ""
        if _ALPHA_DIGIT_ALPHA.search(content):
            return True
    return False


def has_bare_latex(text: str) -> bool:
    """True if text contains LaTeX \\X commands OUTSIDE <math>/<typst>/$...$ regions.

    Such commands can't be reliably converted at render time — preserve the
    original PDF text layer instead of producing broken output.
    """
    if not text:
        return False
    s = _MATH_TYPST_DISPLAY.sub("", text)
    s = _MATH_TYPST_INLINE.sub("", s)
    s = _TYPST_BLOCK.sub("", s)
    s = _DOLLAR_BLOCK_RE.sub("", s)
    s = _DOLLAR_INLINE_RE.sub("", s)
    return bool(re.search(r"\\[a-zA-Z]+", s))


def is_pure_math_text(text: str) -> bool:
    """True if `text` has only math content (no translatable words).

    Strips <math>, <typst>, $...$, LaTeX commands, eq labels, HTML tags,
    then checks for any word-like letter run.
    """
    if not text:
        return False
    s = _MATH_TYPST_DISPLAY.sub("", text)
    s = _MATH_TYPST_INLINE.sub("", s)
    # Do NOT strip <typst>...</typst> blocks — those are renderable layouts.
    s = _DOLLAR_BLOCK_RE.sub("", s)
    s = _DOLLAR_INLINE_RE.sub("", s)
    s = _LATEX_CMD_RE.sub("", s)
    s = _EQ_LABEL_RE.sub("", s)
    s = _HTML_ANY_RE.sub("", s)
    return not _PROSE_LETTER_RUN.search(s)


# Known Typst math identifiers that must NOT be split into separate letters.
_TYPST_MATH_IDENTIFIERS: set[str] = {
    # Greek letters (lowercase + uppercase)
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Zeta",
    "Eta",
    "Theta",
    "Iota",
    "Kappa",
    "Lambda",
    "Mu",
    "Nu",
    "Xi",
    "Omicron",
    "Pi",
    "Rho",
    "Sigma",
    "Tau",
    "Upsilon",
    "Phi",
    "Chi",
    "Psi",
    "Omega",
    # Variant Greek
    "varepsilon",
    "varphi",
    "vartheta",
    "varrho",
    "varsigma",
    # Common math functions
    "frac",
    "sqrt",
    "root",
    "abs",
    "norm",
    "floor",
    "ceil",
    "round",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "arcsin",
    "arccos",
    "arctan",
    "arccot",
    "arcsec",
    "arccsc",
    "sinh",
    "cosh",
    "tanh",
    "coth",
    "sech",
    "csch",
    "log",
    "ln",
    "exp",
    "det",
    "dim",
    "ker",
    "gcd",
    "lcm",
    "max",
    "min",
    "sum",
    "prod",
    "lim",
    "inf",
    "sup",
    "mod",
    "deg",
    "arg",
    # Typst math layout/style
    "vec",
    "mat",
    "cases",
    "binom",
    "display",
    "inline",
    "script",
    "limits",
    "scripts",
    "attach",
    "accent",
    "overline",
    "underline",
    "overbrace",
    "underbrace",
    "cancel",
    "upright",
    "bold",
    "italic",
    "serif",
    "sans",
    "mono",
    "bb",
    "cal",
    "frak",
    # Operator words used in dotted Typst identifiers (plus.minus, minus.plus, etc.)
    # Must be kept intact so the dot-notation survives _split_math_vars.
    "plus",
    "minus",
    "times",
    "div",
    "arrow",
    "tilde",
    "hat",
    "grave",
    "acute",
    "breve",
    "caron",
    "diaer",
    "macron",
    # Unit names — keep intact; LLM should quote them ("rad", "kg") but if bare,
    # splitting into letters is worse than leaving as a multi-letter identifier.
    "rad",
    "radian",
    "radians",
    "rpm",
    "rps",
    "kg",
    "mg",
    "km",
    "cm",
    "mm",
    "ms",
    "ns",
    "hz",
    "khz",
    "mhz",
    "ghz",
    # Relational / logical
    "not",
    "and",
    "or",
    "in",
    "gt",
    "lt",
    "eq",
    "approx",
    "equiv",
    "subset",
    "supset",
    "union",
    "inter",
    "forall",
    "exists",
    "therefore",
    "because",
    # Dots & special symbols (Typst names)
    "dots",
    "dots.c",
    "dots.b",
    "dots.v",
    "dots.down",
    "infty",
    "iint",
    "iiint",
    "oint",
    "int",
    "partial",
    "nabla",
    "ell",
    "hbar",
    "planck",
    "nothing",
    "space",
    "thin",
    "med",
    "thick",
    "circle",
    "ast",
    "star",
    "compose",
    "bullet",
    "without",
    "wr",
    "asymp",
    "prop",
    "models",
    "perp",
    "parallel",
    "bowtie",
    "smile",
    "frown",
    "aleph",
    "wp",
    "Re",
    "Im",
    "empty",
    "surd",
    "top",
    "bot",
    "angle",
    "triangle",
    "backslash",
    "flat",
    "natural",
    "sharp",
    "club",
    "diamond",
    "heart",
    "spade",
    "quad",
    "wide",
    "degree",
    "dot",
    "slash",
    "bar",
    "harpoon",
    "brace",
    "bracket",
    "op",
}
# Also build a pattern that matches a known identifier anchored at the start
# of a word — used for greedy left-to-right tokenisation.
_IDENT_ALPHA = re.compile(r"[a-zA-Z]{2,}")


_LATEX_IDENT_RENAME: dict[str, str] = {
    # Dots
    "cdot": "dot.op",
    "cdots": "dots.c",
    "ldots": "dots",
    "vdots": "dots.v",
    "ddots": "dots.down",
    
    # Fonts
    "mathbf": "bold",
    "mathrm": "upright",
    "mathit": "italic",
    "mathsf": "sans",
    "mathtt": "mono",
    "mathcal": "cal",
    "mathbb": "bb",
    "mathfrak": "frak",
    "boldsymbol": "bold",
    "text": "upright",
    "textbf": "bold",
    "textit": "italic",
    "textrm": "upright",
    "rm": "upright",
    "bf": "bold",
    "it": "italic",
    "operatorname": "upright",

    # Accents
    "vec": "arrow",
    "bar": "macron",
    "check": "caron",
    "ddot": "dot.double",
    "dddot": "dot.triple",
    "ddddot": "dot.quad",
    "mathring": "circle",

    # Operators & Symbols
    "pm": "plus.minus",
    "mp": "minus.plus",
    "times": "times",
    "div": "div",
    "ast": "ast",
    "star": "star",
    "circ": "compose",
    "bullet": "bullet",
    "oplus": "plus.circle",
    "ominus": "minus.circle",
    "otimes": "times.circle",
    "oslash": "div.circle",
    "odot": "dot.circle",
    "cup": "union",
    "cap": "inter",
    "uplus": "union.plus",
    "sqcap": "inter.sq",
    "sqcup": "union.sq",
    "vee": "or",
    "wedge": "and",
    "setminus": "without",
    "wr": "wr",

    # Relations
    "liminf": "liminf",
    "limsup": "limsup",
    "varliminf": "liminf",
    "varlimsup": "limsup",
    "varnothing": "empty",
    "leq": "lt.eq",
    "geq": "gt.eq",
    "neq": "eq.not",
    "le": "lt.eq",
    "ge": "gt.eq",
    "ne": "eq.not",
    "ll": "lt.double",
    "gg": "gt.double",
    "equiv": "equiv",
    "sim": "tilde.op",
    "simeq": "tilde.eq",
    "asymp": "asymp",
    "approx": "approx",
    "cong": "tilde.equiv",
    "doteq": "eq.est",
    "propto": "prop",
    "models": "models",
    "perp": "perp",
    "mid": "bar.v",
    "parallel": "parallel",
    "bowtie": "bowtie",
    "ltimes": "times.l",
    "rtimes": "times.r",
    "smile": "smile",
    "frown": "frown",
    "in": "in",
    "notin": "in.not",
    "ni": "in.rev",
    "subset": "subset",
    "supset": "supset",
    "subseteq": "subset.eq",
    "supseteq": "supset.eq",

    # Arrows
    "leftarrow": "arrow.l",
    "rightarrow": "arrow.r",
    "leftrightarrow": "arrow.l.r",
    "Leftarrow": "arrow.l.double",
    "Rightarrow": "arrow.r.double",
    "Leftrightarrow": "arrow.l.r.double",
    "mapsto": "arrow.r.bar",
    "to": "arrow.r",
    "implies": "arrow.r.double",
    "iff": "arrow.l.r.double",
    "gets": "arrow.l",
    "hookleftarrow": "arrow.l.hook",
    "hookrightarrow": "arrow.r.hook",
    "rightharpoonup": "harpoon.rt",
    "leftharpoonup": "harpoon.lt",
    "rightharpoondown": "harpoon.rb",
    "leftharpoondown": "harpoon.lb",
    "rightleftharpoons": "harpoons.rtlb",

    # Misc
    "aleph": "aleph",
    "wp": "wp",
    "Re": "Re",
    "Im": "Im",
    "emptyset": "empty",
    "nabla": "nabla",
    "surd": "surd",
    "top": "top",
    "bot": "bot",
    "angle": "angle",
    "triangle": "triangle",
    "backslash": "backslash",
    "forall": "forall",
    "exists": "exists",
    "nexists": "exists.not",
    "neg": "not",
    "lnot": "not",
    "flat": "flat",
    "natural": "natural",
    "sharp": "sharp",
    "clubsuit": "club",
    "diamondsuit": "diamond",
    "heartsuit": "heart",
    "spadesuit": "spade",
    "infty": "infty",
    "partial": "partial",
    "quad": "quad",
    "qquad": "wide",
    "O": "O",
    "degree": "degree",

    # Brackets
    "langle": "angle.l",
    "rangle": "angle.r",
    "lbrace": "brace.l",
    "rbrace": "brace.r",
    "lceil": "ceil.l",
    "rceil": "ceil.r",
    "lfloor": "floor.l",
    "rfloor": "floor.r",
    "lbrack": "bracket.l",
    "rbrack": "bracket.r",
}


def _split_math_vars(math_content: str) -> str:
    """Insert spaces between consecutive-letter variable products in Typst math.

    In Typst math, ``bh`` is one identifier. We need ``b h`` (two separate
    variables multiplied implicitly).  Known identifiers like ``frac``, ``sin``,
    ``theta`` etc. are kept intact, as are any word immediately followed by ``(``
    (function-call syntax).
    """

    def _replace(m: re.Match) -> str:
        word = m.group(0)
        # LaTeX identifier with a different Typst name — rename
        if word in _LATEX_IDENT_RENAME:
            return _LATEX_IDENT_RENAME[word]
        # Known Typst math identifier — keep as-is
        if word in _TYPST_MATH_IDENTIFIERS:
            return word
        # Function call (followed by open paren) — keep as-is
        if m.end() < len(math_content) and math_content[m.end()] == "(":
            return word
        # Try greedy left-to-right: peel off known identifiers, then single chars
        result_parts: list[str] = []
        i = 0
        while i < len(word):
            matched = False
            # Try longest known identifier starting at position i
            for length in range(min(len(word) - i, 12), 1, -1):
                candidate = word[i : i + length]
                if candidate in _TYPST_MATH_IDENTIFIERS:
                    result_parts.append(candidate)
                    i += length
                    matched = True
                    break
            if not matched:
                result_parts.append(word[i])
                i += 1
        return " ".join(result_parts)

    return _IDENT_ALPHA.sub(_replace, math_content)


def to_typst_native(text: str) -> str:
    """Convert hybrid HTML/Typst-math text to raw Typst markup string.

    Expects <math> tags to contain Typst math syntax (no backslash LaTeX).
    Output is a Typst markup string suitable for eval(markup, mode: "markup").
    """
    if not text:
        return ""

    # Placeholder for # in our generated function calls — keeps them safe
    # through the user-content escape step (which would otherwise turn # into \#).
    PH = "\x00"
    # Placeholder for raw <typst> blocks — pass through untouched.
    TS_OPEN, TS_CLOSE = "\x02", "\x03"

    result = text

    # 0. Stash raw Typst blocks; restore at the very end to bypass all escaping.
    raw_typst: list[str] = []

    def _stash_typst(m: re.Match) -> str:
        raw_typst.append(m.group(1))
        return f"{TS_OPEN}{len(raw_typst) - 1}{TS_CLOSE}"

    result = _TYPST_BLOCK.sub(_stash_typst, result)

    # 1. Display math → $ content $ (spaces = display/block in Typst).
    #    Inline math → $content$ (no spaces = inline in Typst, stays in text flow).
    #    Also split multi-letter variable products (e.g. bh → b h) inside math.
    result = _MATH_TYPST_DISPLAY.sub(
        lambda m: f"$ {_split_math_vars(m.group(1).strip())} $", result
    )
    result = _MATH_TYPST_INLINE.sub(
        lambda m: f"${_split_math_vars(m.group(1).strip())}$", result
    )

    # 2. Bold / italic / sup / sub → function-call syntax.
    # Function calls avoid word-boundary issues that break `_x_θ`-style emphasis.
    result = _BOLD.sub(lambda m: f"{PH}strong[{m.group(1)}]", result)
    result = _ITALIC.sub(lambda m: f"{PH}emph[{m.group(1)}]", result)
    result = _SUP.sub(lambda m: f"{PH}super[{m.group(1)}]", result)
    result = _SUB.sub(lambda m: f"{PH}sub[{m.group(1)}]", result)

    # 3. Strip remaining unknown HTML tags (preserving < > inside $...$)
    result = _strip_tags_outside_math(result)

    # 4. Escape Typst-special chars in user content (outside math)
    result = _escape_typst_outside_math(result, clean_math=True)

    # 5. Restore # for our generated function calls
    result = result.replace(PH, "#")

    # 6. Restore raw Typst blocks; clean LLM mistakes inside $...$ math regions.
    if raw_typst:

        def _restore(m: re.Match) -> str:
            content = raw_typst[int(m.group(1))]
            content = content.replace("\\/", "/")  # legacy prompt artifact
            # Inside each inline $...$, run full math cleanup (LaTeX → Typst, var split).
            return re.sub(
                r"\$([^$]+?)\$",
                lambda mm: _clean_math_chunk(f"${mm.group(1)}$"),
                content,
                flags=re.DOTALL,
            )

        result = re.sub(f"{TS_OPEN}(\\d+){TS_CLOSE}", _restore, result)

    return result


# ---------------------------------------------------------------------------
# TOC parser
# ---------------------------------------------------------------------------

# Matches: title (non-greedy) + optional dot leaders + page_number
# Lookahead: followed by whitespace+digit (next entry) or end of string.
# Page numbers limited to 1-3 digits to avoid matching years like 2023.
_TOC_BLOB_RE = re.compile(
    r"(.+?)\s*(?:(?:\.\s*){2,})?\s*(\d{1,3})(?=\s+\d|\s*$)",
    re.DOTALL,
)
_DOT_SEQ_RE = re.compile(r"\s*(?:\.\s*){2,}")


def parse_toc_line(line: str) -> tuple[str, str] | None:
    """Extract (title, page_number) from a single TOC line, or None."""
    line = line.strip()
    if not line:
        return None
    m = re.match(r"^(.*?)\s+(\d+)\s*$", line)
    if m:
        return m.group(1).strip(), m.group(2)
    return None


def parse_toc_entries(text: str) -> list[tuple[str, str | None]]:
    """Parse TOC text into (title, page_num) pairs.

    Handles both newline-separated lines and single-line blob format
    (entries concatenated with dot leaders or bare spaces).
    page_num is None for entries where no page number could be found.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        result = []
        for line in lines:
            parsed = parse_toc_line(line)
            result.append(parsed if parsed else (line, None))
        return result

    # Single-line blob: extract via regex, preserving gap text between matches
    entries: list[tuple[str, str | None]] = []
    prev_end = 0
    for m in _TOC_BLOB_RE.finditer(text):
        gap = text[prev_end : m.start()].strip()
        if gap:
            clean = _DOT_SEQ_RE.sub(" ", gap).strip()
            if clean:
                entries.append((clean, None))
        title = m.group(1).strip()
        if title:
            entries.append((title, m.group(2)))
        prev_end = m.end()
    tail = _DOT_SEQ_RE.sub(" ", text[prev_end:]).strip()
    if tail:
        entries.append((tail, None))
    return entries if entries else [(text.strip(), None)]
