"""German number words, used to generate spoken notes and to check grounding."""

import re

# Building blocks of German number words: 0-19 are irregular, tens follow a pattern
_ONES = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
         "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn",
         "achtzehn", "neunzehn"]
_TENS = {2: "zwanzig", 3: "dreißig", 4: "vierzig", 5: "fünfzig", 6: "sechzig", 7: "siebzig",
         8: "achtzig", 9: "neunzig"}


def de_words(n: int) -> str:
    """Write an integer as one German word.

    Input:  n - integer from 0 to 999
    Output: e.g. 21 -> "einundzwanzig", 132 -> "hundertzweiunddreißig"
    """
    # 0-19: look up directly
    if n < 20:
        return _ONES[n]
    # 20-99: German says the ones first ("ein-und-zwanzig"); "eins" becomes "ein"
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return _TENS[tens]
        return ("ein" if ones == 1 else _ONES[ones]) + "und" + _TENS[tens]
    # 100-999: "hundert" or "<n>hundert", followed by the rest (recursive call)
    hundreds, rest = divmod(n, 100)
    head = "hundert" if hundreds == 1 else _ONES[hundreds] + "hundert"
    return head + (de_words(rest) if rest else "")


# Reverse lookup word -> number for 0-999, built once with de_words()
_WORD_TO_INT = {de_words(n): n for n in range(1000)}
_WORD_TO_INT.update({"ein" + de_words(n): n for n in range(100, 200)})  # "einhundertzwanzig"
_WORD_TO_INT["eins"] = 1
_WORD_TO_INT["ein"] = 1


def numbers_in(text: str) -> list[float]:
    """Find all numbers mentioned in a text, written as digits or as German words.

    Input:  text, e.g. "Temperatur siebenunddreißig komma fünf, RR 130 zu 80"
    Output: list of numbers, e.g. [130.0, 80.0, 37.0, 37.5, 5.0]
            (parts of a decimal are listed too; callers only check membership)
    """
    # Digits, with "," or "." as decimal separator: "37,5" -> 37.5
    found = [float(m.replace(",", ".")) for m in re.findall(r"\d+(?:[.,]\d+)?", text)]
    # Number words: split into lower-case words and look each one up
    tokens = re.findall(r"[a-zäöüß]+", text.lower())
    for i, tok in enumerate(tokens):
        if tok not in _WORD_TO_INT:
            continue
        value = float(_WORD_TO_INT[tok])
        found.append(value)
        # Spoken decimals: "<number> komma <digit>" -> e.g. 37 + 5/10 = 37.5
        if i + 2 < len(tokens) and tokens[i + 1] == "komma" and tokens[i + 2] in _WORD_TO_INT:
            found.append(value + _WORD_TO_INT[tokens[i + 2]] / 10)
    return found
