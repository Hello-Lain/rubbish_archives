"""Pattern-compatible noun singularization used by the SHIELD CHAIR scorer."""

from __future__ import annotations

import re

_PREPOSITIONS = {
    "about", "before", "during", "of", "till", "above", "behind", "except",
    "off", "to", "across", "below", "for", "on", "under", "after", "beneath",
    "from", "onto", "until", "among", "beside", "in", "out", "unto", "around",
    "besides", "into", "over", "upon", "at", "between", "near", "since", "with",
    "athwart", "betwixt", "beyond", "but", "by",
}

_RULES = [
    (r"(?i)(.)ae$", r"\1a"),
    (r"(?i)(.)itis$", r"\1itis"),
    (r"(?i)(.)eaux$", r"\1eau"),
    (r"(?i)(quiz)zes$", r"\1"),
    (r"(?i)(matr)ices$", r"\1ix"),
    (r"(?i)(ap|vert|ind)ices$", r"\1ex"),
    (r"(?i)^(ox)en", r"\1"),
    (r"(?i)(alias|status)es$", r"\1"),
    (r"(?i)([octop|vir])i$", r"\1us"),
    (r"(?i)(cris|ax|test)es$", r"\1is"),
    (r"(?i)(shoe)s$", r"\1"),
    (r"(?i)(o)es$", r"\1"),
    (r"(?i)(bus)es$", r"\1"),
    (r"(?i)([m|l])ice$", r"\1ouse"),
    (r"(?i)(x|ch|ss|sh)es$", r"\1"),
    (r"(?i)(m)ovies$", r"\1ovie"),
    (r"(?i)(.)ombies$", r"\1ombie"),
    (r"(?i)(s)eries$", r"\1eries"),
    (r"(?i)([^aeiouy]|qu)ies$", r"\1y"),
    (r"([aeo]l)ves$", r"\1f"),
    (r"([^d]ea)ves$", r"\1f"),
    (r"arves$", "arf"),
    (r"erves$", "erve"),
    (r"([nlw]i)ves$", r"\1fe"),
    (r"(?i)([lr])ves$", r"\1f"),
    (r"([aeo])ves$", r"\1ve"),
    (r"(?i)(sive)s$", r"\1"),
    (r"(?i)(tive)s$", r"\1"),
    (r"(?i)(hive)s$", r"\1"),
    (r"(?i)([^f])ves$", r"\1fe"),
    (r"(?i)(^analy)ses$", r"\1sis"),
    (r"(?i)((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$", r"\1\2sis"),
    (r"(?i)(.)opses$", r"\1opsis"),
    (r"(?i)(.)yses$", r"\1ysis"),
    (r"(?i)(h|d|r|o|n|b|cl|p)oses$", r"\1ose"),
    (r"(?i)(fruct|gluc|galact|lact|ket|malt|rib|sacchar|cellul)ose$", r"\1ose"),
    (r"(?i)(.)oses$", r"\1osis"),
    (r"(?i)([ti])a$", r"\1um"),
    (r"(?i)(n)ews$", r"\1ews"),
    (r"(?i)s$", ""),
]
_COMPILED_RULES = [(re.compile(pattern), replacement) for pattern, replacement in _RULES]

_UNINFLECTED = {
    "bison", "debris", "headquarters", "pincers", "trout", "bream", "diabetes",
    "herpes", "pliers", "tuna", "breeches", "djinn", "high-jinks", "proceedings",
    "whiting", "britches", "eland", "homework", "rabies", "wildebeest", "carp",
    "elk", "innings", "salmon", "chassis", "flounder", "jackanapes", "scissors",
    "christmas", "gallows", "mackerel", "series", "clippers", "georgia", "measles",
    "shears", "cod", "graffiti", "mews", "species", "contretemps", "mumps", "swine",
    "corps", "news", "swiss",
}
_UNCOUNTABLE = {
    "advice", "equipment", "happiness", "luggage", "news", "software", "bread",
    "fruit", "information", "mathematics", "progress", "understanding", "butter",
    "furniture", "ketchup", "mayonnaise", "research", "water", "cheese", "garbage",
    "knowledge", "meat", "rice", "electricity", "gravel", "love", "mustard", "sand",
}
_IE = {
    "alergie", "cutie", "hoagie", "newbie", "softie", "veggie", "auntie", "doggie",
    "hottie", "nightie", "sortie", "weenie", "beanie", "eyrie", "indie", "oldie",
    "stoolie", "yuppie", "birdie", "freebie", "junkie", "^pie", "sweetie", "zombie",
    "bogie", "goonie", "laddie", "pixie", "techie", "bombie", "groupie", "laramie",
    "quickie", "^tie", "collie", "hankie", "lingerie", "reverie", "toughie", "cookie",
    "hippie", "meanie", "rookie", "valkyrie",
}
_IRREGULAR = {
    "atlantes": "atlas", "atlases": "atlas", "axes": "axe", "beeves": "beef",
    "brethren": "brother", "children": "child", "corpora": "corpus",
    "corpuses": "corpus", "ephemerides": "ephemeris", "feet": "foot",
    "ganglia": "ganglion", "geese": "goose", "genera": "genus", "genii": "genie",
    "graffiti": "graffito", "helves": "helve", "kine": "cow", "leaves": "leaf",
    "loaves": "loaf", "men": "man", "mongooses": "mongoose", "monies": "money",
    "moves": "move", "mythoi": "mythos", "numena": "numen", "occipita": "occiput",
    "octopodes": "octopus", "opera": "opus", "opuses": "opus", "our": "my",
    "oxen": "ox", "penes": "penis", "penises": "penis", "people": "person",
    "sexes": "sex", "soliloquies": "soliloquy", "teeth": "tooth", "testes": "testis",
    "trilbys": "trilby", "turves": "turf", "zoa": "zoon",
}


def singularize(word: str, custom: dict[str, str] | None = None) -> str:
    """Singularize one word using the subset used by Pattern's CHAIR scorer."""

    custom = custom or {}
    if word in custom:
        return custom[word]
    if "-" in word:
        parts = word.split("-")
        if len(parts) > 1 and parts[1] in _PREPOSITIONS:
            return singularize(parts[0], custom) + "-" + "-".join(parts[1:])
    if word.endswith("'"):
        return singularize(word[:-1], custom) + "'s"
    lower = word.lower()
    if any(value.endswith(lower) for value in _UNINFLECTED | _UNCOUNTABLE):
        return word
    if any(lower.endswith(value + "s") for value in _IE):
        return lower
    for inflected, singular in _IRREGULAR.items():
        if lower.endswith(inflected):
            return re.sub(f"(?i){inflected}$", singular, word)
    for suffix, replacement in _COMPILED_RULES:
        match = suffix.search(word)
        if match:
            resolved = replacement
            for index, group in enumerate(match.groups(), 1):
                if group is None:
                    resolved = resolved.replace(f"\\{index}", "")
            return suffix.sub(resolved, word)
    return word
