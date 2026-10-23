"""
Text normaliser shared by every decision type.

It rewrites number words and durations into one canonical form, so that
"fourteen days", "two weeks" and "14 days" read the same, and so do
"a month" / "30 days", "six months" / "180 days", "a year" / "365 days" and
"an hour" / "60 minutes". The data uses exactly these conventions (checked on
the grounded=true pairs), and without them the encoder has to learn unit
arithmetic from a few hundred examples - that was the main source of confident
`grounded` errors. Business days are left alone: they do not convert.

Applied to every field so the shared backbone always sees one vocabulary.
Pure regex, a few microseconds per field.
"""

import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_tens_alt = "|".join(_TENS)
_units_alt = "|".join(sorted(_UNITS, key=len, reverse=True))
_below100 = r"(?:(?:%s)(?:[\s-](?:%s))?|(?:%s))" % (_tens_alt, _units_alt, _units_alt)

# "three hundred and sixty-five", "one hundred and eighty", "a hundred", "three hundred"
_HUNDREDS = re.compile(
    r"\b(?P<h>a|%s)\s+hundred(?:\s+(?:and\s+)?(?P<rest>%s))?\b" % (_units_alt, _below100), re.IGNORECASE)
_NUM_WORD = re.compile(
    r"\b(?:(?P<tens>%s)(?:[\s-](?P<tail>%s))?|(?P<unit>%s))\b" % (_tens_alt, _units_alt, _units_alt),
    re.IGNORECASE)


def _below100_value(s: str) -> int:
    m = _NUM_WORD.fullmatch(s.strip())
    if not m:
        return 0
    if m.group("tens"):
        return _TENS[m.group("tens").lower()] + (_UNITS[m.group("tail").lower()] if m.group("tail") else 0)
    return _UNITS[m.group("unit").lower()]


def _hundreds_sub(m: re.Match) -> str:
    h = m.group("h").lower()
    v = (1 if h == "a" else _UNITS[h]) * 100
    if m.group("rest"):
        v += _below100_value(m.group("rest"))
    return str(v)


def _num_sub(m: re.Match) -> str:
    if m.group("tens"):
        v = _TENS[m.group("tens").lower()]
        if m.group("tail"):
            v += _UNITS[m.group("tail").lower()]
        return str(v)
    return str(_UNITS[m.group("unit").lower()])


_A_UNIT = re.compile(r"\b(?:a|an|one|a single)\s+(minute|hour|day|week|month|year|fortnight)\b", re.IGNORECASE)
_HALF_HOUR = re.compile(r"\bhalf an hour\b", re.IGNORECASE)
_QUARTER_HOUR = re.compile(r"\b(?:a\s+)?quarter of an hour\b", re.IGNORECASE)
_DURATION = re.compile(
    r"\b(\d+)\s+(business days?|working days?|minutes?|hours?|days?|weeks?|fortnights?|months?|years?)\b",
    re.IGNORECASE)
_TO_BASE = {"minute": ("minutes", 1), "hour": ("minutes", 60),
            "day": ("days", 1), "week": ("days", 7), "fortnight": ("days", 14),
            "month": ("days", 30), "year": ("days", 365)}
_THOUSAND = re.compile(r"\b(\d+)\s+thousand\b", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


def _duration_sub(m: re.Match) -> str:
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith(("business", "working")):
        return "%d business days" % n
    base = unit[:-1] if unit.endswith("s") else unit
    name, mult = _TO_BASE[base]
    return "%d %s" % (n * mult, name)


def normalize(text: str) -> str:
    if not text:
        return ""
    t = _HUNDREDS.sub(_hundreds_sub, text)
    t = _HALF_HOUR.sub("30 minutes", t)
    t = _QUARTER_HOUR.sub("15 minutes", t)
    t = _NUM_WORD.sub(_num_sub, t)
    t = _A_UNIT.sub(lambda m: "1 " + m.group(1).lower(), t)
    t = re.sub(r"\ba thousand\b", "1000", t, flags=re.IGNORECASE)
    t = _THOUSAND.sub(lambda m: str(int(m.group(1)) * 1000), t)
    t = _DURATION.sub(_duration_sub, t)
    return _SPACES.sub(" ", t).strip()


if __name__ == "__main__":
    for s in ["We tell monthly subscribers two weeks before any change",
              "the link holds for fourteen days, twenty-five seats, a week",
              "You have thirty minutes; lockout after ten failed attempts, one days",
              "a fortnight, half an hour, one hundred requests, 1 day, an hour",
              "three hundred and sixty-five days from delivery; one hundred and eighty days",
              "you have a quarter of an hour to use it, half an hour",
              "a month, two months, six months, a year, the a year warranty, 3-5 business days",
              "one thousand requests per minute, 1 thousand, a thousand",
              "three hundred requests per minute; 48 hours; Rates are reviewed every 2 days months"]:
        print(normalize(s))
