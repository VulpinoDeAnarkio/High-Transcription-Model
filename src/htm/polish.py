"""Polish 1st-person gender cues as a weak speaker prior — not meeting-specific."""

from __future__ import annotations

import re

_MASC = re.compile(
    r"\b("
    r"powiedziałem|zdążyłem|chciałem|mogłem|zrobiłem|poszedłem|byłem|"
    r"dostałem|zostałem|widziałem|myślałem|wiedziałem|przyszedłem|"
    r"napisałem|usłyszałem|zrozumiałem"
    r")\b",
    re.IGNORECASE,
)
_FEM = re.compile(
    r"\b("
    r"powiedziałam|zdążyłam|chciałam|chciałabym|mogłam|mogłabym|"
    r"zrobiłam|poszłam|byłam|dostałam|zostałam|widziałam|myślałam|"
    r"wiedziałam|przyszłam|napisałam|usłyszałam|zrozumiałam"
    r")\b",
    re.IGNORECASE,
)


def first_person_gender(text: str) -> str | None:
    """Return 'M', 'K' or None from Polish 1sg verb morphology."""
    t = text or ""
    m = len(_MASC.findall(t))
    f = len(_FEM.findall(t))
    if m and not f:
        return "M"
    if f and not m:
        return "K"
    return None
