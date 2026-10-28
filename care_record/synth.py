"""Synthetic care notes with known ground truth.

Approach: sample a ground-truth record first, then render every fact into a spoken-style
German phrase (filler words, abbreviations, numbers as digits or words) and join the
phrases into a transcript. Because each phrase is inserted verbatim, its text is also the
ground-truth `evidence`. Generation is template-based and seeded, so the dataset is
reproducible and contains no real patient data.

Each case also lists `expect_review`: paths of items a careful system should flag for
human review (hedged values, contradictions, implausible numbers). For these items the
evaluation only requires the entity to be found, not the exact value.
"""

import json
import random
from pathlib import Path

from .numbers_de import de_words
from .schema import ExtractedNote

PATIENTS = [
    "Frau Berger", "Herr Lindner", "Frau Kowalski", "Herr Yilmaz", "Frau Hoffmann",
    "Herr Brandt", "Frau Neumann", "Herr Petersen", "Frau Schreiber", "Herr Albrecht",
    "Frau Krüger", "Herr Vogel", "Frau Seidel", "Herr Hartmann", "Frau Engel",
]
SHIFTS = ["Frühdienst", "Spätdienst", "Nachtdienst"]
FILLERS = ["Ähm, ", "Also, ", "Dann noch, ", "Ach ja, ", "Joa, ", "Und ", "Äh, ", ""]
CLOSERS = ["Das war's.", "Ja, das wär's von mir.", "Sonst nichts Besonderes.", "Okay, Ende.", ""]


def say_int(n: int, words: bool) -> str:
    """Say an integer the way a caregiver might dictate it.

    Input:  n, words (True = as German words)
    Output: e.g. "130" or "hundertdreißig"
    """
    return de_words(n) if words else str(n)


def say_decimal(x: float, words: bool) -> str:
    """Say a number with one decimal, German style.

    Input:  x, words (True = as German words)
    Output: 38.5 -> '38,5' or 'achtunddreißig komma fünf'; 38.0 -> '38'
    """
    # Split into whole part and first decimal digit (38.5 -> 38, 5)
    whole, frac = divmod(round(x * 10), 10)
    if words:
        return de_words(whole) + (f" komma {de_words(frac)}" if frac else "")
    return f"{whole},{frac}" if frac else str(whole)


# ---------------------------------------------------------------- fact renderers
# Each renderer returns (truth_item_dict, spoken_phrase). evidence == spoken_phrase.


def vital(rng: random.Random, kind: str, words: bool, value=None, diastolic=None):
    """Create one vital sign: a random realistic value and a spoken phrase for it.

    Input:  rng (seeded random generator), kind (e.g. "pulse"), words (numbers as words?),
            value/diastolic (optional fixed values instead of random ones)
    Output: (truth item dict, spoken phrase), e.g. ({"kind": "pulse", "value": 72}, "Puls lag bei 72")
    """
    # Each kind: pick a value in a normal range, then one of several ways to say it
    if kind == "blood_pressure":
        s = value or rng.randrange(105, 170, 5)
        d = diastolic or rng.randrange(60, 95, 5)
        phrase = rng.choice([
            "Blutdruck {s} zu {d}", "RR {s} zu {d}", "RR war {s} zu {d}",
            "Blutdruck gemessen, {s} zu {d}", "der Blutdruck lag bei, äh, {s} zu {d}",
        ]).format(s=say_int(s, words), d=say_int(d, words))
        return {"kind": kind, "value": s, "diastolic": d}, phrase
    if kind == "pulse":
        v = value or rng.randint(58, 98)
        phrase = rng.choice(["Puls {v}", "Puls lag bei {v}", "Herzfrequenz {v}", "Puls war {v}"])
        return {"kind": kind, "value": v}, phrase.format(v=say_int(v, words))
    if kind == "temperature":
        v = value or rng.choice([36.4, 36.6, 36.8, 37.1, 37.4, 37.9, 38.2, 38.5])
        phrase = rng.choice(["Temperatur {v}", "Temperatur {v} Grad", "Fieber gemessen, {v} Grad",
                             "Temp war {v}"])
        return {"kind": kind, "value": v}, phrase.format(v=say_decimal(v, words))
    if kind == "oxygen_saturation":
        v = value or rng.randint(89, 99)
        phrase = rng.choice(["Sättigung {v} Prozent", "SpO2 {v}", "Sauerstoffsättigung {v} Prozent"])
        return {"kind": kind, "value": v}, phrase.format(v=say_int(v, words))
    if kind == "blood_glucose":
        v = value or rng.randint(85, 260)
        phrase = rng.choice(["BZ {v}", "Blutzucker {v}", "Blutzucker gemessen, {v}"])
        return {"kind": kind, "value": v}, phrase.format(v=say_int(v, words))
    if kind == "respiratory_rate":
        v = value or rng.randint(12, 24)
        phrase = rng.choice(["Atemfrequenz {v}", "Atmung {v} pro Minute"])
        return {"kind": kind, "value": v}, phrase.format(v=say_int(v, words))
    if kind == "weight":
        v = value or rng.randint(48, 95)
        phrase = rng.choice(["gewogen, {v} Kilo", "Gewicht {v} Kilo"])
        return {"kind": kind, "value": v}, phrase.format(v=say_int(v, words))
    raise ValueError(kind)


# (name, dose, spoken forms with dose, spoken form without dose)
MEDS = [
    ("Ramipril", "5 mg", ["Ramipril fünf Milligramm", "Ramipril 5 mg"], "das Ramipril"),
    ("Metformin", "500 mg", ["Metformin fünfhundert", "Metformin 500"], "Metformin"),
    ("Pantoprazol", "40 mg", ["Pantoprazol vierzig", "Panto 40"], "das Pantoprazol"),
    ("L-Thyroxin", "50 µg", ["L-Thyroxin fünfzig Mikrogramm", "L-Thyroxin 50"], "das L-Thyroxin"),
    ("ASS", "100 mg", ["ASS hundert", "ASS 100"], "ASS"),
    ("Torasemid", "10 mg", ["Torasemid zehn Milligramm", "Torasemid 10 mg"], "das Torasemid"),
    ("Novaminsulfon", "500 mg", ["Novaminsulfon fünfhundert", "Novalgin 500"], "Novaminsulfon"),
    ("Bisoprolol", "2,5 mg", ["Bisoprolol zwei komma fünf", "Bisoprolol 2,5"], "Bisoprolol"),
    ("Simvastatin", "20 mg", ["Simvastatin zwanzig", "Simva 20"], "das Simvastatin"),
    ("Melperon", "25 mg", ["Melperon fünfundzwanzig", "Melperon 25 mg"], "Melperon"),
    ("Insulin", "8 IE", ["acht Einheiten Insulin", "Insulin 8 IE"], "Insulin"),
    ("Macrogol", "1 Beutel", ["einen Beutel Macrogol", "Macrogol ein Beutel"], "Macrogol"),
]
# Oral tablets only: used where the phrase mentions "die Tablette".
TABLET_MEDS = [m for m in MEDS if m[0] not in ("Insulin", "Macrogol")]
# Spoken brand names that map to a canonical name in the truth.
MED_ALIASES = {"Novalgin": "Novaminsulfon", "Panto": "Pantoprazol", "Simva": "Simvastatin"}


def medication(rng: random.Random, status: str, med=None):
    """Create one medication entry and a spoken phrase for it.

    Input:  rng, status ("given" or "refused"), med (optional fixed entry from MEDS)
    Output: (truth item dict, spoken phrase), e.g.
            ({"name": "Ramipril", "dose": "5 mg", "status": "given"}, "Ramipril 5 mg gegeben")
    """
    name, dose, with_dose, without_dose = med or rng.choice(MEDS)
    # 70 % of the time the dose is spoken; otherwise it is not, so the truth has no dose either
    if rng.random() < 0.7:
        spoken = rng.choice(with_dose)
    else:
        spoken, dose = without_dose, None
    # Wrap the drug in a sentence that says whether it was given or refused
    if status == "given":
        phrase = rng.choice(["{m} gegeben", "{m} hat sie genommen", "hat {m} ohne Probleme genommen",
                             "{m} verabreicht"])
    else:
        phrase = rng.choice(["{m} hat sie verweigert", "wollte {m} nicht nehmen",
                             "{m} abgelehnt, hat's ausgespuckt", "{m} verweigert"])
    return {"name": name, "dose": dose, "status": status}, phrase.format(m=spoken)


INCIDENTS = {
    "fall": [
        ("Sturz im Bad, keine sichtbaren Verletzungen",
         "ist heute im Bad gestürzt, keine sichtbaren Verletzungen, Arzt informiert"),
        ("Sturz neben dem Bett aufgefunden",
         "lag morgens neben dem Bett auf dem Boden, hat sich wohl rausrollen lassen"),
        ("Sturz beim Transfer, Hämatom am Arm",
         "ist beim Transfer weggerutscht, hat jetzt 'nen blauen Fleck am Arm"),
    ],
    "pain": [
        ("Knieschmerzen", "klagt über Schmerzen im Knie"),
        ("Rückenschmerzen", "hat wieder Rückenschmerzen"),
        ("Kopfschmerzen", "hatte nachmittags Kopfschmerzen"),
    ],
    "wound": [
        ("Dekubitus am Steiß, Verbandswechsel", "Dekubitus am Steiß, Verband gewechselt, sieht etwas besser aus"),
        ("Hautriss am Unterarm", "kleiner Hautriss am Unterarm, mit Pflaster versorgt"),
        ("Rötung an der Ferse", "Ferse rechts gerötet, hab sie weich gelagert"),
    ],
    "other": [
        ("Erbrechen", "hat sich nach dem Mittagessen einmal übergeben"),
        ("Unruhe in der Nacht", "war nachts sehr unruhig, ist mehrfach aufgestanden"),
    ],
}


def incident(rng: random.Random, kind: str, words: bool):
    """Create one incident and a spoken phrase for it.

    Input:  rng, kind ("fall", "pain", "wound", "other"), words (pain score as words?)
    Output: (truth item dict, spoken phrase)
    """
    description, phrase = rng.choice(INCIDENTS[kind])
    item = {"type": kind, "description": description, "pain_score": None}
    # Pain sometimes comes with a score on the 0-10 scale, which then must be extracted too
    if kind == "pain" and rng.random() < 0.7:
        score = rng.randint(2, 8)
        phrase += rng.choice([", Schmerzskala {s}", ", auf der Skala so {s} von zehn", ", NRS {s}"]).format(
            s=say_int(score, words))
        item["pain_score"] = score
    return item, phrase


OBSERVATIONS = {
    "kognition_kommunikation": [
        ("zeitlich desorientiert", "war heute zeitlich nicht orientiert, wusste nicht welcher Tag ist"),
        ("gesprächig, gut ansprechbar", "war gut drauf, sehr gesprächig"),
        ("wirkte verwirrt, hat Personal nicht erkannt", "wirkte verwirrt, hat mich nicht erkannt"),
    ],
    "mobilitaet_beweglichkeit": [
        ("mit Rollator mobil", "ist mit dem Rollator über den Flur gelaufen"),
        ("Unterstützung beim Transfer Bett-Rollstuhl",
         "musste beim Transfer vom Bett in den Rollstuhl unterstützt werden"),
        ("alle zwei Stunden umgelagert", "alle zwei Stunden umgelagert"),
    ],
    "krankheitsbezogene_anforderungen": [
        ("Kompressionsstrümpfe angezogen", "Kompressionsstrümpfe angezogen"),
        ("Ödeme an beiden Unterschenkeln", "Beine waren unten ziemlich dick, Ödeme beidseits"),
        ("Atemnot bei Belastung", "hatte beim Laufen etwas Luftnot"),
    ],
    "selbstversorgung": [
        ("reduzierte Nahrungsaufnahme, Mittagessen zur Hälfte",
         "hat beim Mittagessen nur die Hälfte gegessen"),
        ("geringe Trinkmenge, ca. 800 ml", "Trinkmenge heute nur so achthundert Milliliter"),
        ("Grundpflege am Waschbecken selbstständig", "hat sich am Waschbecken selber gewaschen"),
        ("Inkontinenzmaterial gewechselt", "Inkoeinlage gewechselt, war nass"),
    ],
    "soziale_beziehungen": [
        ("Besuch der Tochter", "Tochter war zu Besuch"),
        ("Teilnahme an der Singgruppe", "war bei der Singgruppe dabei"),
        ("zieht sich zurück, wollte im Zimmer bleiben", "wollte heut nicht raus, ist lieber im Zimmer geblieben"),
    ],
    "wohnen_haeuslichkeit": [
        ("wünscht offenes Fenster nachts", "will nachts das Fenster offen haben"),
        ("Zimmer aufgeräumt, Wäsche einsortiert", "Zimmer aufgeräumt und die Wäsche eingeräumt"),
    ],
}


def observation(rng: random.Random, area: str | None = None):
    """Create one observation for an SIS area and a spoken phrase for it.

    Input:  rng, area (optional; random if None)
    Output: (truth item dict, spoken phrase)
    """
    area = area or rng.choice(list(OBSERVATIONS))
    text, phrase = rng.choice(OBSERVATIONS[area])
    return {"area": area, "text": text}, phrase


# ---------------------------------------------------------------- case builder


class CaseBuilder:
    """Collects (section, truth_item, phrase) and assembles transcript + truth.

    Usage: add() every fact of a case, then build() returns the finished case.
    """

    def __init__(self, rng: random.Random, case_id: str, category: str):
        """Input: rng (seeded random generator), case_id (e.g. "case_001"), category name."""
        self.rng, self.case_id, self.category = rng, case_id, category
        self.parts: list[tuple[str | None, dict | None, str, bool]] = []  # (section, item, phrase, review)

    def add(self, section: str | None, item: dict | None, phrase: str, review: bool = False):
        """Add one spoken phrase to the case.

        Input:  section - where the item belongs in the truth, or None for a phrase that
                          must NOT produce an item (e.g. "Blutdruck gemessen, trag ich nach")
                item    - truth item dict (None if section is None)
                phrase  - what the caregiver says
                review  - True if a careful system must flag this item for human review
        Output: none
        """
        self.parts.append((section, item, phrase, review))

    def build(self, shuffle: bool = True) -> dict:
        """Assemble the transcript and the matching ground truth.

        Input:  shuffle - put the phrases in random order (like real, unordered dictation)
        Output: {"id", "category", "transcript", "truth", "expect_review"}
        """
        rng = self.rng
        parts = self.parts[:]
        if shuffle:
            rng.shuffle(parts)
        # Opening sentence with a (fictional) resident name
        patient = rng.choice(PATIENTS)
        opener = rng.choice([
            f"So, {patient}, {rng.choice(SHIFTS)}.",
            f"Also, {patient}, Zimmer {say_int(rng.randint(1, 40), rng.random() < 0.5)}.",
            f"Übergabe {patient}.",
        ])
        sentences = [opener]
        truth = {"observations": [], "vital_signs": [], "medications": [], "incidents": []}
        expect_review = []
        for section, item, phrase, review in parts:
            # Spoken style: a random filler word in front ("Ähm, ", "Also, ", ...)
            filler = rng.choice(FILLERS)
            shown = phrase if filler else phrase[0].upper() + phrase[1:]
            sentences.append(f"{filler}{shown}.")
            if section is not None:
                # The phrase is in the transcript word for word, so it is also the true evidence
                truth[section].append({**item, "evidence": shown, "uncertain": review})
                if review:
                    expect_review.append(f"{section}[{len(truth[section]) - 1}]")
        closer = rng.choice(CLOSERS)
        if closer:
            sentences.append(closer)
        truth["summary"] = ""
        ExtractedNote.model_validate(truth)  # generator must only produce valid truth
        return {
            "id": self.case_id,
            "category": self.category,
            "transcript": " ".join(sentences),
            "truth": truth,
            "expect_review": expect_review,
        }


VITAL_KINDS = ["blood_pressure", "pulse", "temperature", "oxygen_saturation", "blood_glucose",
               "respiratory_rate", "weight"]


def _standard_content(b: CaseBuilder, words: bool | None = None, n_vitals=(1, 3), n_meds=(0, 2),
                      n_incidents=(0, 1), n_obs=(1, 3)):
    """Add a random mix of ordinary facts to a case.

    Input:  b - the CaseBuilder to add to
            words - numbers as words (True), as digits (False), or mixed 30/70 (None)
            n_* - (min, max) number of vital signs, medications, incidents and observations
    Output: none (adds phrases to b)
    """
    rng = b.rng
    # w() decides per number whether it is spoken as words
    w = (lambda: rng.random() < 0.3) if words is None else (lambda: words)
    for kind in rng.sample(VITAL_KINDS[:5], rng.randint(*n_vitals)):
        b.add("vital_signs", *vital(rng, kind, w()))
    for med in rng.sample(MEDS, rng.randint(*n_meds)):
        b.add("medications", *medication(rng, "given" if rng.random() < 0.8 else "refused", med))
    for kind in rng.sample(list(INCIDENTS), rng.randint(*n_incidents)):
        b.add("incidents", *incident(rng, kind, w()))
    for area in rng.sample(list(OBSERVATIONS), rng.randint(*n_obs)):
        b.add("observations", *observation(rng, area))


def make_case(rng: random.Random, case_id: str, category: str) -> dict:
    """Create one case of a given category; each category tests one kind of difficulty.

    Input:  rng, case_id, category (a key of CATEGORY_COUNTS)
    Output: finished case dict (see CaseBuilder.build)
    """
    b = CaseBuilder(rng, case_id, category)

    if category == "standard":
        _standard_content(b)

    elif category == "nothing_noteworthy":
        b.add(None, None, rng.choice([
            "alles ruhig, keine Besonderheiten",
            "nichts Besonderes, hat gut geschlafen",
            "unauffälliger Dienst, keine Vorkommnisse",
        ]))

    elif category == "missing_value":
        # Measurement mentioned, value not stated -> must NOT appear in the record.
        b.add(None, None, rng.choice([
            "Blutdruck hab ich gemessen, aber den Wert hab ich nicht mehr im Kopf, trag ich nach",
            "Blutzucker müsste noch jemand messen, bin nicht mehr dazu gekommen",
            "Puls hab ich gemessen, war normal",  # "normal" is not a number
            "Temperatur wollte sie nicht messen lassen",
        ]))
        _standard_content(b, n_vitals=(0, 1), n_meds=(0, 1), n_incidents=(0, 0), n_obs=(1, 2))

    elif category == "self_correction":
        kind = rng.choice(["pulse", "blood_glucose", "oxygen_saturation"])
        item, phrase = vital(rng, kind, False)
        offsets = [-6, -4] if kind == "oxygen_saturation" else [-6, -4, 4, 6]
        wrong = int(item["value"]) + rng.choice(offsets)
        phrase = phrase.replace(str(int(item["value"])), f"{wrong}, nee, Quatsch, {int(item['value'])}")
        b.add("vital_signs", item, phrase)
        _standard_content(b, n_vitals=(0, 1), n_meds=(0, 1), n_incidents=(0, 1), n_obs=(1, 2))

    elif category == "spoken_numbers":
        _standard_content(b, words=True, n_vitals=(2, 4), n_incidents=(0, 1))
        # make sure there is at least one pain score in words
        b.add("incidents", *incident(rng, "pain", True))

    elif category == "refused_meds":
        for med in rng.sample(MEDS, 2):
            b.add("medications", *medication(rng, "refused", med))
        _standard_content(b, n_vitals=(0, 2), n_meds=(0, 1), n_incidents=(0, 1), n_obs=(1, 2))

    elif category == "hedged":
        # Vague value -> extract it, but flag as uncertain.
        v = rng.choice([37.5, 38.0, 38.5])
        b.add("vital_signs", {"kind": "temperature", "value": v},
              rng.choice(["Temperatur war so um die {v}, glaub ich", "hatte vielleicht so {v} Grad, "
                          "hab's nicht genau gesehen"]).format(v=say_decimal(v, False)), review=True)
        _standard_content(b, n_vitals=(0, 1), n_meds=(0, 1), n_incidents=(0, 0), n_obs=(1, 2))

    elif category == "contradiction":
        # Two statements that conflict -> one item, flagged for review.
        name = rng.choice(TABLET_MEDS)[0]
        b.add("medications", {"name": name, "dose": None, "status": "given"},
              f"{name} hab ich gegeben, also, die Tablette lag später aber noch auf dem Nachttisch",
              review=True)
        b.add("incidents", {"type": "fall", "description": "möglicher Sturz, unklar", "pain_score": None},
              "kein Sturz heute, wobei, sie saß vorhin auf dem Boden vorm Bett", review=True)
        _standard_content(b, n_vitals=(1, 2), n_meds=(0, 0), n_incidents=(0, 0), n_obs=(0, 1))

    elif category == "implausible":
        # Number that is physiologically implausible (e.g. a dictation slip) -> flag it.
        item, phrase = rng.choice([
            ({"kind": "temperature", "value": 3.8}, "Temperatur drei komma acht"),
            ({"kind": "pulse", "value": 720}, "Puls 720"),
            ({"kind": "oxygen_saturation", "value": 9}, "Sättigung 9 Prozent"),
        ])
        b.add("vital_signs", item, phrase, review=True)
        _standard_content(b, n_vitals=(0, 1), n_meds=(0, 1), n_incidents=(0, 0), n_obs=(1, 2))

    else:
        raise ValueError(category)

    return b.build()


CATEGORY_COUNTS = {
    "standard": 16,
    "nothing_noteworthy": 4,
    "missing_value": 4,
    "self_correction": 4,
    "spoken_numbers": 4,
    "refused_meds": 4,
    "hedged": 3,
    "contradiction": 3,
    "implausible": 3,
}


def generate_dataset(seed: int = 42) -> list[dict]:
    """Create the full synthetic dataset.

    Input:  seed - same seed, same dataset (42 = dev set, 1042 = test set)
    Output: list of cases, CATEGORY_COUNTS[category] cases per category
    """
    rng = random.Random(seed)
    cases, n = [], 0
    for category, count in CATEGORY_COUNTS.items():
        for _ in range(count):
            n += 1
            cases.append(make_case(rng, f"case_{n:03d}", category))
    return cases


def write_jsonl(cases: list[dict], path: Path) -> None:
    """Write cases as JSON lines (one case per line). Input: cases, path. Output: none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON lines file. Input: path. Output: list of dicts (empty lines are skipped)."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
