"""Format explicit floor sections without changing address components."""

import re


FLOOR_LABELS = {
    # Labels observed in the frozen development set.
    "english": re.compile(
        r"\b(?:Lower Ground Floor|Ground Floor|Upper Floors?)\s*[:：︰]",
        re.IGNORECASE,
    ),
    "chinese": re.compile(r"(?:地下低層|地下|樓上各層)\s*[:：︰]"),
}
SEPARATORS = {"english": ", ", "chinese": "，"}
BOUNDARY = ",，;；。"
JOINED_LABEL = re.compile(r"(?:\b(?:and|or)\b|[&＆及和與与或])$", re.IGNORECASE)
FLOOR_RULE_REVISION = "floor-sections-v2"


def format_floor_sections(address: str, language: str) -> str:
    matches = list(FLOOR_LABELS[language].finditer(address))
    if len(matches) < 2 or matches[0].start() != 0:
        return address

    edits = []
    for i, current in enumerate(matches[1:], start=1):
        previous = matches[i - 1]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(address)
        before = address[previous.end():current.start()]
        after = address[current.end():end]
        body = before.rstrip()
        if (not body.strip(BOUNDARY + " ") or not after.strip(BOUNDARY + " ")
                or len(body) == len(before) or body[-1] in BOUNDARY
                or JOINED_LABEL.search(body)):
            continue
        edits.append((previous.end() + len(body), current.start()))

    for start, stop in reversed(edits):
        address = address[:start] + SEPARATORS[language] + address[stop:]
    return address
