"""German number words, used to generate spoken notes and to check grounding."""

import re

_ONES = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
         "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn",
         "achtzehn", "neunzehn"]
_TENS = {2: "zwanzig", 3: "dreißig", 4: "vierzig", 5: "fünfzig", 6: "sechzig", 7: "siebzig",
         8: "achtzig", 9: "neunzig"}


def de_words(n: int) -> str:
    """Integer 0..999 as German words, e.g. 132 -> 'hundertzweiunddreißig'."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return _TENS[tens]
        return ("ein" if ones == 1 else _ONES[ones]) + "und" + _TENS[tens]
    hundreds, rest = divmod(n, 100)
    head = "hundert" if hundreds == 1 else _ONES[hundreds] + "hundert"
    return head + (de_words(rest) if rest else "")


_WORD_TO_INT = {de_words(n): n for n in range(1000)}
_WORD_TO_INT.update({"ein" + de_words(n): n for n in range(100, 200)})  # "einhundertzwanzig"
_WORD_TO_INT["eins"] = 1
_WORD_TO_INT["ein"] = 1


def numbers_in(text: str) -> list[float]:
    """All numbers mentioned in text, as digits ("37,5") or words ("siebenunddreißig komma fünf")."""
    found = [float(m.replace(",", ".")) for m in re.findall(r"\d+(?:[.,]\d+)?", text)]
    tokens = re.findall(r"[a-zäöüß]+", text.lower())
    for i, tok in enumerate(tokens):
        if tok not in _WORD_TO_INT:
            continue
        value = float(_WORD_TO_INT[tok])
        found.append(value)
        if i + 2 < len(tokens) and tokens[i + 1] == "komma" and tokens[i + 2] in _WORD_TO_INT:
            found.append(value + _WORD_TO_INT[tokens[i + 2]] / 10)
    return found
