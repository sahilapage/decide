"""
Text normaliser shared by every decision type.

It rewrites number words and common durations into one canonical form, so
"fourteen days", "two weeks" and "14 days" all read as "14 days". This matters
most for `grounded`, where the wrong drafts usually differ from the article in
one number, but it is applied to every field so the backbone always sees the
same vocabulary.

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

# "twenty-five", "twenty five", "twenty", "fourteen", "a hundred", "one hundred"
_NUM_WORD = re.compile(
    r"\b(?:(?P<tens>%s)(?:[\s-](?P<tail>%s))?|(?P<unit>%s))\b" % (_tens_alt, _units_alt, _units_alt),
    re.IGNORECASE,
)
_HUNDRED = re.compile(r"\b(?:a|one)\s+hundred\b", re.IGNORECASE)


def _num_sub(m: re.Match) -> str:
    if m.group("tens"):
        v = _TENS[m.group("tens").lower()]
        if m.group("tail"):
            v += _UNITS[m.group("tail").lower()]
        return str(v)
    return str(_UNITS[m.group("unit").lower()])


_A_UNIT = re.compile(r"\b(?:a|an|one|a single)\s+(minute|hour|day|week|month|year|fortnight)\b", re.IGNORECASE)
_WEEKS = re.compile(r"\b(\d+)\s+weeks?\b", re.IGNORECASE)
_FORTNIGHT = re.compile(r"\b(\d+)\s+fortnights?\b", re.IGNORECASE)
_HALF_HOUR = re.compile(r"\bhalf an hour\b", re.IGNORECASE)
_PLURAL = re.compile(r"\b(\d+)\s+(minute|hour|day|month|year|business day|working day)s?\b", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    if not text:
        return ""
    t = _HUNDRED.sub("100", text)
    t = _HALF_HOUR.sub("30 minutes", t)
    t = _NUM_WORD.sub(_num_sub, t)
    t = _A_UNIT.sub(lambda m: "1 " + m.group(1).lower(), t)
    t = _FORTNIGHT.sub(lambda m: "%d days" % (int(m.group(1)) * 14), t)
    t = _WEEKS.sub(lambda m: "%d days" % (int(m.group(1)) * 7), t)
    # one canonical plural form: "1 days" / "3 minute" -> "1 days" / "3 minutes"
    t = _PLURAL.sub(lambda m: "%s %ss" % (m.group(1), m.group(2).lower()), t)
    return _SPACES.sub(" ", t).strip()


if __name__ == "__main__":
    for s in ["We tell monthly subscribers two weeks before any change",
              "the link holds for fourteen days, twenty-five seats, a week",
              "You have thirty minutes; lockout after ten failed attempts, one days",
              "a fortnight, half an hour, one hundred requests, 1 day"]:
        print(normalize(s))
