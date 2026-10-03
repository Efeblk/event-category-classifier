"""Apply a few explicit request boundaries before learned classification."""

import re
import unicodedata

# These words identify explicit activity mentions. They do not cover all meanings.
PATTERNS = {
    "concert": r"\b(?:konser\w*|concert\w*|gig\w*)\b|\b(?:live|canli) (?:music|muzik|jazz|caz)\b",
    "theatre": r"\b(?:tiyatro\w*|theatre\w*|theater\w*)\b|\b(?:stage play|stage drama\w*|sahne oyunu)\b|\b(?:bir|a|the) (?:oyun|play)\b",
    "stand_up": r"\bstand[ -]?up\b|\b(?:comedian\w*|komedyen\w*)\b|\bcomedy show\b",
}
BEFORE_NEGATION = re.compile(r"\b(?:no|not|never|without|avoid|except|don't|dont|skip)\b")
AFTER_NEGATION = re.compile(r"\b(?:istem\w*|olmas\w*|degil|haric|yerine)\b|\b(?:is not|isn't|isnt)\b")


def plain_text(text):
    text = text.lower().translate(str.maketrans({"\u0131": "i"}))
    return "".join(char for char in unicodedata.normalize("NFD", text) if not unicodedata.combining(char))


def inspect_request(text):
    """Return explicit positive categories and a model input without exclusions."""
    plain = plain_text(text)
    matches = sorted((match.start(), match.end(), label)
                     for label, pattern in PATTERNS.items()
                     for match in re.finditer(pattern, plain))
    positives, negatives = set(), []
    previous_negative = False
    for index, (start, end, label) in enumerate(matches):
        previous = matches[index - 1][1] if index else 0
        following = matches[index + 1][0] if index + 1 < len(matches) else len(plain)
        before = plain[max(previous, start - 32):start]
        after = plain[end:min(following, end + 40)]
        # A shared exclusion also applies to a coordinated list: "not theatre or stand-up".
        shared_exclusion = previous_negative and re.fullmatch(r"\s*(?:or|and|ve|veya|ya da|,)\s*", before)
        negative = bool(BEFORE_NEGATION.search(before) or AFTER_NEGATION.search(after) or shared_exclusion)
        if negative:
            negatives.append((start, end))
        else:
            positives.add(label)
        previous_negative = negative
    for start, end in reversed(negatives):
        plain = plain[:start] + " " * (end - start) + plain[end:]
    return positives, bool(negatives), plain


def guard_request(text):
    positives, has_exclusions, cleaned = inspect_request(text)
    plain = plain_text(text)
    player = re.search(r"\b(?:spotify|youtube|playlist|calma listesi)\b", plain)
    player_command = re.search(r"\b(?:play|stream|ac|cal|oynat)\b", plain)
    attendance = re.search(r"\b(?:attend|ticket\w*|bilet\w*|gitmek|gide\w*)\b", plain)
    direct_playback = re.search(r"\bplay (?:\w+\s+){0,3}(?:music|album|song|playlist)\b|\bmuzik (?:ac\w*|cal\w*|oynat\w*)\b", plain)
    if (player and player_command or direct_playback) and not attendance:
        return text, "music_player_command"
    if re.search(r"\b(?:sinema\w*|film\w*|movie\w*|cinema\w*)\b", plain) and not positives:
        return text, "unsupported_activity"
    if len(positives) > 1:
        return text, "multiple_categories"
    if has_exclusions and not positives:
        # A different activity can be implicit. Let the learned model inspect the rest.
        if re.search(r"\b(?:ama|but|instead|yerine)\b|,", cleaned):
            return cleaned, None
        return text, "only_exclusions"
    # Use the original text unless we removed an explicit excluded category.
    return cleaned if has_exclusions else text, None


def keyword_baseline(text):
    _, reason = guard_request(text)
    positives, _, _ = inspect_request(text)
    return next(iter(positives)) if not reason and len(positives) == 1 else "unclear"


def classify_input(artifact, text):
    from classifier import _validated_model, classify_request

    if not isinstance(text, str) or not text.strip() or len(text) > 5000:
        raise ValueError("Enter a request with 1 to 5,000 characters.")
    # Validate the artifact even when an input guard can decide the outcome.
    _validated_model(artifact)
    cleaned, reason = guard_request(text)
    if reason:
        return {"label": "unclear", "model": artifact["model_name"],
                "reason": reason, "decision_source": "input_guard"}
    result = classify_request(artifact, cleaned)
    result["model"] = artifact["model_name"]
    result["decision_source"] = "model"
    return result
