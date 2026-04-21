#!/usr/bin/env python3
"""
한국어 발음 커버리지 매트릭스 분석기.

검증 범주:
- 7종성(ㄱ/ㄴ/ㄷ/ㄹ/ㅁ/ㅂ/ㅇ) x 종결/연음/절음
- 된소리(ㄲ/ㄸ/ㅃ/ㅆ/ㅉ)
- 외래어/영문 혼용
- 고유수사 + 단위명사
- 전화번호 / 계좌번호
- 긴 숫자
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FINAL_LABELS = ("ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ")
POSITION_LABELS = ("종결", "연음", "절음")
TENSE_LABELS = ("ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ")
COUNTER_PATTERN = re.compile(
    r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|열한|열두|열세|열네|열다섯|"
    r"스무|스물|서른|마흔|쉰|예순|일흔|여든|아흔)\s*"
    r"(개|명|마리|시간|시|번째|잔|병|권|대|살|층|달|개월|주|통|장)\b"
)
PHONE_PATTERN = re.compile(r"(?<!\d)0\d{1,2}[- ]\d{3,4}[- ]\d{4}(?!\d)")
ACCOUNT_PATTERN = re.compile(r"(?<!\d)\d{2,4}[- ]\d{2,6}[- ]\d{2,6}(?!\d)")
LONG_NUMBER_PATTERN = re.compile(r"(?<!\d)\d[\d,]{4,}(?!\d)")
ASCII_PATTERN = re.compile(r"[A-Za-z]")
LOANWORD_PATTERN = re.compile(
    r"(아이폰|와이파이|유에스비|에이아이|블루투스|앱|메일|코드|링크|서버|클라우드)"
)

CHOOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
JONGSEONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

FINAL_GROUP_MAP = {
    "ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㄳ": "ㄱ", "ㅋ": "ㄱ",
    "ㄴ": "ㄴ", "ㄵ": "ㄴ", "ㄶ": "ㄴ",
    "ㄷ": "ㄷ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ",
    "ㄹ": "ㄹ", "ㄺ": "ㄹ", "ㄻ": "ㄹ", "ㄼ": "ㄹ", "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㄿ": "ㄹ", "ㅀ": "ㄹ",
    "ㅁ": "ㅁ",
    "ㅂ": "ㅂ", "ㅄ": "ㅂ", "ㅍ": "ㅂ",
    "ㅇ": "ㅇ",
}


def is_hangul_syllable(char: str) -> bool:
    return 0xAC00 <= ord(char) <= 0xD7A3


def decompose_syllable(char: str) -> tuple[str, str]:
    code = ord(char) - 0xAC00
    cho_idx = code // 588
    jong_idx = code % 28
    return CHOOSEONG[cho_idx], JONGSEONG[jong_idx]


def next_hangul_index(text: str, start: int) -> int | None:
    for idx in range(start, len(text)):
        if is_hangul_syllable(text[idx]):
            return idx
    return None


def classify_position(text: str, idx: int, final_group: str) -> str:
    next_idx = next_hangul_index(text, idx + 1)
    if next_idx is None:
        return "종결"

    between = text[idx + 1:next_idx]
    if any(ch.isspace() for ch in between):
        return "종결"
    if any(ch in ".!?,:;" for ch in between):
        return "종결"
    if between.strip() and not all(ch in "\"'()[]{}" for ch in between):
        return "종결"

    next_cho, _ = decompose_syllable(text[next_idx])
    if next_cho == "ㅇ":
        return "연음"
    return "절음"


def analyze(texts: list[str]) -> dict:
    final_counts = {
        final: {position: 0 for position in POSITION_LABELS}
        for final in FINAL_LABELS
    }
    tense_counts = {label: 0 for label in TENSE_LABELS}
    foreign_mixed = 0
    native_counter = 0
    phone_or_account = 0
    long_numbers = 0

    for text in texts:
        if not text.strip():
            continue

        if ASCII_PATTERN.search(text) or LOANWORD_PATTERN.search(text):
            foreign_mixed += 1
        if COUNTER_PATTERN.search(text):
            native_counter += 1
        if PHONE_PATTERN.search(text) or ACCOUNT_PATTERN.search(text):
            phone_or_account += 1
        if LONG_NUMBER_PATTERN.search(text):
            long_numbers += 1

        for tense in TENSE_LABELS:
            tense_counts[tense] += text.count(tense)

        for idx, char in enumerate(text):
            if not is_hangul_syllable(char):
                continue
            cho, jong = decompose_syllable(char)
            if cho in tense_counts:
                tense_counts[cho] += 1
            if not jong:
                continue
            final_group = FINAL_GROUP_MAP.get(jong)
            if final_group not in final_counts:
                continue
            position = classify_position(text, idx, final_group)
            final_counts[final_group][position] += 1

    final_targets = {
        final: {position: 1 for position in POSITION_LABELS}
        for final in FINAL_LABELS
    }
    tense_targets = {label: 3 for label in TENSE_LABELS}

    missing: list[str] = []
    for final in FINAL_LABELS:
        for position in POSITION_LABELS:
            if final_counts[final][position] < final_targets[final][position]:
                missing.append(f"종성 {final} / {position}")
    for tense in TENSE_LABELS:
        if tense_counts[tense] < tense_targets[tense]:
            missing.append(f"된소리 {tense}")
    if foreign_mixed < 3:
        missing.append("외래어/영문 혼용")
    if native_counter < 3:
        missing.append("고유수사+단위명사")
    if phone_or_account < 2:
        missing.append("전화번호/계좌번호")
    if long_numbers < 2:
        missing.append("긴 숫자")

    total_slots = len(FINAL_LABELS) * len(POSITION_LABELS) + len(TENSE_LABELS) + 4
    satisfied_slots = total_slots - len(missing)

    return {
        "sentence_count": len([text for text in texts if text.strip()]),
        "final_counts": final_counts,
        "final_targets": final_targets,
        "tense_counts": tense_counts,
        "tense_targets": tense_targets,
        "foreign_mixed": foreign_mixed,
        "native_counter": native_counter,
        "phone_or_account": phone_or_account,
        "long_numbers": long_numbers,
        "missing": missing,
        "coverage_percent": round((satisfied_slots / total_slots) * 100, 1),
    }


def report(result: dict) -> str:
    lines: list[str] = []
    lines.append(f"문장 수: {result['sentence_count']}")
    lines.append(f"커버리지: {result['coverage_percent']}%")
    lines.append("")
    lines.append("[종성 7종 x 위치]")
    for final in FINAL_LABELS:
        counts = result["final_counts"][final]
        summary = ", ".join(
            f"{position}={counts[position]}/{result['final_targets'][final][position]}"
            for position in POSITION_LABELS
        )
        lines.append(f"- {final}: {summary}")

    lines.append("")
    lines.append("[된소리 5종]")
    for tense in TENSE_LABELS:
        lines.append(
            f"- {tense}: {result['tense_counts'][tense]}/{result['tense_targets'][tense]}"
        )

    lines.append("")
    lines.append("[기타 범주]")
    lines.append(f"- 외래어/영문 혼용: {result['foreign_mixed']}")
    lines.append(f"- 고유수사+단위명사: {result['native_counter']}")
    lines.append(f"- 전화번호/계좌번호: {result['phone_or_account']}")
    lines.append(f"- 긴 숫자: {result['long_numbers']}")

    if result["missing"]:
        lines.append("")
        lines.append("[부족한 범주]")
        for item in result["missing"]:
            lines.append(f"- {item}")
    else:
        lines.append("")
        lines.append("모든 필수 범주가 최소 기준을 충족했습니다.")

    return "\n".join(lines)


def load_texts_from_input(value: str) -> list[str]:
    path = Path(value)
    if path.exists():
        return path.read_text(encoding="utf-8").splitlines()
    return [value]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python korean_coverage_matrix.py recording_script_ko.txt")
        return 1

    texts = load_texts_from_input(" ".join(argv[1:]))
    result = analyze(texts)
    print(report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
