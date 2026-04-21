#!/usr/bin/env python3
"""
한국어 TTS 입력을 위한 룰 기반 정규화.

주요 처리:
- 숫자 + 단위 / 통화
- 날짜 / 시간
- 전화번호
- 외래어 / 영문 혼용

예시:
  python normalize_korean.py "2026년 4월 19일 오후 3시 20분에 010-1234-5678로 전화 주세요."
"""

from __future__ import annotations

import re
import sys
from typing import Callable

SINO_DIGITS = ["영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
PHONE_DIGITS = ["공", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
SINO_UNITS = ["", "십", "백", "천"]
LARGE_UNITS = ["", "만", "억", "조"]

NATIVE_CARDINAL = {
    1: "하나",
    2: "둘",
    3: "셋",
    4: "넷",
    5: "다섯",
    6: "여섯",
    7: "일곱",
    8: "여덟",
    9: "아홉",
    10: "열",
    11: "열하나",
    12: "열둘",
    13: "열셋",
    14: "열넷",
    15: "열다섯",
    16: "열여섯",
    17: "열일곱",
    18: "열여덟",
    19: "열아홉",
    20: "스물",
    30: "서른",
    40: "마흔",
    50: "쉰",
    60: "예순",
    70: "일흔",
    80: "여든",
    90: "아흔",
}

COUNTER_FORMS = {
    "하나": "한",
    "둘": "두",
    "셋": "세",
    "넷": "네",
    "스물": "스무",
}

COUNTER_UNITS = (
    "개", "명", "마리", "잔", "병", "권", "대", "채", "살", "층",
    "시", "시간", "번째", "번", "달", "개월", "주", "장", "통", "줄",
)

SPECIAL_WORDS = {
    "wifi": "와이파이",
    "wi-fi": "와이파이",
    "usb": "유에스비",
    "ai": "에이아이",
    "api": "에이피아이",
    "app": "앱",
    "sms": "에스엠에스",
    "otp": "오티피",
    "url": "유알엘",
    "cpu": "씨피유",
    "gpu": "지피유",
}

LETTER_NAMES = {
    "A": "에이",
    "B": "비",
    "C": "씨",
    "D": "디",
    "E": "이",
    "F": "에프",
    "G": "지",
    "H": "에이치",
    "I": "아이",
    "J": "제이",
    "K": "케이",
    "L": "엘",
    "M": "엠",
    "N": "엔",
    "O": "오",
    "P": "피",
    "Q": "큐",
    "R": "알",
    "S": "에스",
    "T": "티",
    "U": "유",
    "V": "브이",
    "W": "더블유",
    "X": "엑스",
    "Y": "와이",
    "Z": "지",
}


def strip_number(raw: str) -> int:
    return int(raw.replace(",", "").strip())


def sino_number(num: int) -> str:
    if num == 0:
        return "영"

    parts: list[str] = []
    unit_index = 0
    while num > 0:
        chunk = num % 10000
        if chunk:
            chunk_text = sino_under_10000(chunk)
            large = LARGE_UNITS[unit_index]
            parts.append(f"{chunk_text}{large}")
        num //= 10000
        unit_index += 1
    return "".join(reversed(parts))


def sino_under_10000(num: int) -> str:
    result: list[str] = []
    digits = list(map(int, f"{num:04d}"))
    for idx, digit in enumerate(digits):
        if digit == 0:
            continue
        power = 3 - idx
        if digit == 1 and power > 0:
            result.append(SINO_UNITS[power])
        else:
            result.append(f"{SINO_DIGITS[digit]}{SINO_UNITS[power]}")
    return "".join(result)


def native_number(num: int) -> str:
    if num <= 0:
        return sino_number(num)
    if num in NATIVE_CARDINAL:
        return NATIVE_CARDINAL[num]
    tens = (num // 10) * 10
    ones = num % 10
    if tens in NATIVE_CARDINAL and ones in NATIVE_CARDINAL:
        return f"{NATIVE_CARDINAL[tens]}{NATIVE_CARDINAL[ones]}"
    return sino_number(num)


def native_counter_number(num: int) -> str:
    text = native_number(num)
    return COUNTER_FORMS.get(text, text)


def digits_to_words(value: str, digit_map: list[str] = PHONE_DIGITS) -> str:
    return "".join(digit_map[int(ch)] for ch in value if ch.isdigit())


def replace_with_pattern(pattern: str, repl: Callable[[re.Match[str]], str], text: str) -> str:
    return re.sub(pattern, repl, text)


def normalize_phone(match: re.Match[str]) -> str:
    groups = [group for group in match.groups() if group]
    return "에 ".join(digits_to_words(group) for group in groups)


def normalize_date(match: re.Match[str]) -> str:
    year = sino_number(strip_number(match.group("year")))
    month = sino_number(strip_number(match.group("month")))
    day = sino_number(strip_number(match.group("day")))
    return f"{year} 년 {month} 월 {day} 일"


def normalize_year(match: re.Match[str]) -> str:
    year = sino_number(strip_number(match.group("year")))
    return f"{year} 년"


def normalize_time(match: re.Match[str]) -> str:
    meridiem = match.group("meridiem") or ""
    hour = int(match.group("hour"))
    minute = match.group("minute")
    second = match.group("second")
    hour_text = native_counter_number(hour) if 1 <= hour <= 12 else sino_number(hour)
    parts = [part for part in [meridiem, f"{hour_text} 시"] if part]
    if minute:
        parts.append(f"{sino_number(int(minute))} 분")
    if second:
        parts.append(f"{sino_number(int(second))} 초")
    return " ".join(parts)


def normalize_prefixed_currency(match: re.Match[str]) -> str:
    symbol = match.group("symbol")
    amount = sino_number(strip_number(match.group("amount")))
    unit = {"₩": "원", "$": "달러", "€": "유로"}[symbol]
    return f"{amount} {unit}"


def normalize_suffixed_currency(match: re.Match[str]) -> str:
    amount = sino_number(strip_number(match.group("amount")))
    unit = match.group("unit")
    return f"{amount} {unit}"


def normalize_counter(match: re.Match[str]) -> str:
    number = int(match.group("amount").replace(",", ""))
    counter = match.group("counter")
    text = native_counter_number(number)
    return f"{text} {counter}"


def normalize_plain_number(match: re.Match[str]) -> str:
    return sino_number(strip_number(match.group("amount")))


def normalize_spelled_word(match: re.Match[str]) -> str:
    word = match.group(0)
    lower = word.lower()
    if lower in SPECIAL_WORDS:
        return SPECIAL_WORDS[lower]
    if re.fullmatch(r"[A-Za-z]+", word):
        return "".join(LETTER_NAMES.get(ch.upper(), ch) for ch in word)
    return word


def normalize(text: str) -> str:
    normalized = text.strip()

    normalized = replace_with_pattern(
        r"(?<!\d)(?P<year>\d{4})\s*년\s*(?P<month>\d{1,2})\s*월\s*(?P<day>\d{1,2})\s*일(?!\d)",
        normalize_date,
        normalized,
    )
    normalized = replace_with_pattern(
        r"(?<!\d)(?P<year>\d{4})\s*년",
        normalize_year,
        normalized,
    )
    normalized = replace_with_pattern(
        (
            r"(?<!\d)(?P<meridiem>오전|오후)?\s*"
            r"(?P<hour>\d{1,2})\s*시"
            r"(?:\s*(?P<minute>\d{1,2})\s*분)?"
            r"(?:\s*(?P<second>\d{1,2})\s*초)?"
        ),
        normalize_time,
        normalized,
    )
    normalized = replace_with_pattern(
        r"(?<!\d)(0\d{1,2})[- ](\d{3,4})[- ](\d{4})(?!\d)",
        normalize_phone,
        normalized,
    )
    normalized = replace_with_pattern(
        r"(?P<symbol>[₩$€])\s*(?P<amount>\d[\d,]*)",
        normalize_prefixed_currency,
        normalized,
    )
    normalized = replace_with_pattern(
        r"(?P<amount>\d[\d,]*)(?P<unit>원|달러|유로)\b",
        normalize_suffixed_currency,
        normalized,
    )

    counter_pattern = "|".join(sorted(COUNTER_UNITS, key=len, reverse=True))
    normalized = replace_with_pattern(
        rf"(?P<amount>\d[\d,]*)(?P<counter>{counter_pattern})\b",
        normalize_counter,
        normalized,
    )

    normalized = replace_with_pattern(
        r"(?<![A-Za-z])[A-Za-z][A-Za-z-]*(?![A-Za-z])",
        normalize_spelled_word,
        normalized,
    )
    normalized = replace_with_pattern(
        r"(?<![\w])(?P<amount>\d[\d,]*)(?![\w])",
        normalize_plain_number,
        normalized,
    )

    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.:;!?])", r"\1", normalized)
    return normalized.strip()


def run_tests() -> None:
    cases = {
        "5,000원": "오천 원",
        "3개": "세 개",
        "2026년": "이천이십육 년",
        "2026년 4월 19일": "이천이십육 년 사 월 십구 일",
        "오후 3시 20분": "오후 세 시 이십 분",
        "010-1234-5678": "공일공에 일이삼사에 오육칠팔",
        "USB": "유에스비",
        "WiFi": "와이파이",
        "AI": "에이아이",
        "₩5,000": "오천 원",
        "$10": "십 달러",
    }
    for raw, expected in cases.items():
        actual = normalize(raw)
        assert actual == expected, f"{raw!r}: {actual!r} != {expected!r}"
    print(f"테스트 통과: {len(cases)}개")


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        print(normalize(" ".join(argv[1:])))
        return 0
    run_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
