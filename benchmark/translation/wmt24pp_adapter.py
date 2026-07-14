"""WMT24++ → `doc` adapter (the crux: reuse the translation core unchanged).

Loads a WMT24++ language-pair file (`en-<xx>.jsonl`), filters bad sources, groups
segments by `document_id`, and wraps each document into the exact `doc` dict that
`translate_document` consumes. After translation, extracts the hypothesis per segment
aligned 1:1 with the reference.

No dependency on `datasets` for loading: the jsonl is fetched directly with httpx
(already a project dep) and cached locally. `datasets` is only needed if you want to
enumerate all configs via `list_all_pairs()`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger("benchmark.translation.adapter")

HF_BASE = "https://huggingface.co/datasets/google/wmt24pp/resolve/main"
CACHE_DIR = Path(__file__).parent / ".cache_wmt24pp"

# Target-locale -> full English language name (interpolated into the prompt).
# Best-effort cover of the WMT24++ 55 locales; validate against
# `datasets.get_dataset_config_names("google/wmt24pp")` before a full sweep.
LOCALE_NAME: dict[str, str] = {
    "ar_EG": "Egyptian Arabic", "ar_SA": "Arabic", "bg_BG": "Bulgarian",
    "bn_IN": "Bengali", "ca_ES": "Catalan", "cs_CZ": "Czech", "da_DK": "Danish",
    "de_DE": "German", "el_GR": "Greek", "es_MX": "Mexican Spanish",
    "et_EE": "Estonian", "fa_IR": "Persian", "fi_FI": "Finnish",
    "fil_PH": "Filipino", "fr_CA": "Canadian French", "fr_FR": "French",
    "gu_IN": "Gujarati", "he_IL": "Hebrew", "hi_IN": "Hindi", "hr_HR": "Croatian",
    "hu_HU": "Hungarian", "id_ID": "Indonesian", "is_IS": "Icelandic",
    "it_IT": "Italian", "ja_JP": "Japanese", "kn_IN": "Kannada", "ko_KR": "Korean",
    "lt_LT": "Lithuanian", "lv_LV": "Latvian", "ml_IN": "Malayalam",
    "mr_IN": "Marathi", "nl_NL": "Dutch", "no_NO": "Norwegian", "pa_IN": "Punjabi",
    "pl_PL": "Polish", "pt_BR": "Brazilian Portuguese", "pt_PT": "Portuguese",
    "ro_RO": "Romanian", "ru_RU": "Russian", "sk_SK": "Slovak", "sl_SI": "Slovenian",
    "sr_RS": "Serbian", "sv_SE": "Swedish", "sw_KE": "Swahili", "sw_TZ": "Swahili",
    "ta_IN": "Tamil",
    "te_IN": "Telugu", "th_TH": "Thai", "tr_TR": "Turkish", "uk_UA": "Ukrainian",
    "ur_PK": "Urdu", "vi_VN": "Vietnamese", "zh_CN": "Simplified Chinese",
    "zh_TW": "Traditional Chinese", "zu_ZA": "Zulu",
}

# Minimal fallback for an unmapped locale's base subtag.
_BASE_NAME = {
    "ar": "Arabic", "zh": "Chinese", "pt": "Portuguese", "fr": "French",
    "es": "Spanish", "en": "English",
}


def language_name(locale: str) -> str:
    """Full language name for a target locale like 'vi_VN'. Falls back to the base
    subtag name, else the locale string itself (with a warning)."""
    if locale in LOCALE_NAME:
        return LOCALE_NAME[locale]
    base = locale.split("_")[0]
    if base in _BASE_NAME:
        return _BASE_NAME[base]
    logger.warning("No language name for locale %r; using it verbatim in the prompt.", locale)
    return locale


@dataclass
class SegRecord:
    """One aligned segment: source, reference, and where it came from."""

    pair: str          # e.g. "en-vi_VN"
    document_id: str
    seg_index: int     # order within the document (== internal task id)
    source: str
    reference: str     # WMT24++ post-edit `target`
    domain: str


@dataclass
class DocBundle:
    """All segments of one WMT24++ document + the built `doc` dict."""

    pair: str
    document_id: str
    segs: list[SegRecord]
    doc: dict = field(default_factory=dict)


def pair_config(target_locale: str) -> str:
    """WMT24++ config/file stem for an English->target pair."""
    return f"en-{target_locale}"


def _download_jsonl(pair: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{pair}.jsonl"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{HF_BASE}/{pair}.jsonl"
    logger.info("Downloading %s", url)
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dest


def load_pair(
    target_locale: str,
    *,
    drop_bad_source: bool = True,
) -> list[DocBundle]:
    """Load one pair, filter bad sources, group into documents (order preserved).

    Returns a list of DocBundle, each with its `doc` dict ready for translate_document.
    """
    pair = pair_config(target_locale)
    path = _download_jsonl(pair)
    tgt_name = language_name(target_locale)

    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n_total = len(rows)
    if drop_bad_source:
        rows = [r for r in rows if not r.get("is_bad_source", False)]
    logger.info("%s: %d rows (%d after is_bad_source filter)", pair, n_total, len(rows))

    # Group by document_id, preserving first-seen order and within-doc order.
    docs: dict[str, list[dict]] = {}
    for r in rows:
        docs.setdefault(str(r.get("document_id", "0")), []).append(r)

    bundles: list[DocBundle] = []
    for doc_id, seg_rows in docs.items():
        segs = [
            SegRecord(
                pair=pair,
                document_id=doc_id,
                seg_index=i,
                source=sr["source"],
                reference=sr.get("target", sr.get("original_target", "")),
                domain=sr.get("domain", "unknown"),
            )
            for i, sr in enumerate(seg_rows)
        ]
        doc = {
            "source_language": "English",
            "target_language": tgt_name,
            "pages": [{"elements": [
                {"category": "TEXT", "source_text": s.source} for s in segs
            ]}],
        }
        bundles.append(DocBundle(pair=pair, document_id=doc_id, segs=segs, doc=doc))
    return bundles


def extract_hypotheses(bundle: DocBundle) -> list[dict]:
    """After translate_document mutated bundle.doc in place, read back the hypothesis
    per element (aligned 1:1 with segments) and pair it with the reference.

    `collect_translatables` walks pages->elements in order and writes into
    element['translated_text'], so element[i] corresponds to segs[i].
    """
    elements = bundle.doc["pages"][0]["elements"]
    out: list[dict] = []
    for seg, elem in zip(bundle.segs, elements):
        hyp = elem.get("translated_text", "")
        out.append({
            "pair": seg.pair,
            "document_id": seg.document_id,
            "seg_index": seg.seg_index,
            "domain": seg.domain,
            "source": seg.source,
            "reference": seg.reference,
            "hypothesis": hyp,
            # pipeline falls back to source when it can't translate an id.
            "is_fallback": bool(hyp) and hyp == seg.source,
            "is_empty": not hyp,
        })
    return out


def list_all_pairs() -> list[str]:
    """All target locales available in google/wmt24pp (needs `datasets`)."""
    try:
        from datasets import get_dataset_config_names
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "list_all_pairs() needs `datasets`; or pass locales explicitly from LOCALE_NAME."
        ) from exc
    configs = get_dataset_config_names("google/wmt24pp")
    return [c.split("en-", 1)[1] for c in configs if c.startswith("en-")]
