"""A small high-precision intent baseline, written from train requests only."""
import re
import unicodedata
from classifier import normalize_text


def ascii_text(text):
    text = normalize_text(text).replace("ı", "i")
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))

# Each tuple requires every pattern. Multiple matching intents abstain.
RULES = {
    "alarm_set": ((r"\balarm\w*", r"\b(kur|ayarla|olustur)\w*"), (r"\buyandir\w*",)),
    "weather_query": ((r"\bhava\w*", r"\b(durum|nasil|tahmin)\w*"), (r"\b(yagmur|kar)\b", r"\b(var|yag|olacak)\w*")),
    "play_music": ((r"\b(muzik|sarki|caz|playlist)\w*", r"\b(cal|oynat|dinle|baslat)\w*"),),
    "recommendation_events": ((r"\betkinlik\w*", r"\b(goster|bul|oner|hangi|var)\w*"),),
    "calendar_set": ((r"\btakvim\w*", r"\b(ekle|ayarla|olustur)\w*"), (r"\btoplanti\w*", r"\b(ayarla|programla)\w*")),
    "datetime_query": ((r"\bsaat\b", r"\bkac\b"), (r"\b(tarih|gunlerden)\w*", r"\b(ne|nedir)\b")),
    "iot_hue_lightoff": ((r"\b(isik|lamba)\w*", r"\b(kapat|kapans)\w*"),),
}


def parse_request(text):
    text = ascii_text(text)
    if re.search(r"\b(degil|istemiyorum|yapma|calma|kurma|kapatma)\b", text):
        return "unclear"
    matches = [label for label, alternatives in RULES.items()
               if any(all(re.search(pattern, text) for pattern in patterns) for patterns in alternatives)]
    return matches[0] if len(matches) == 1 else "unclear"


def parser_scores(actual, predicted):
    answered = [i for i, label in enumerate(predicted) if label != "unclear"]
    correct = sum(a == b for a,b in zip(actual, predicted))
    return {"coverage": len(answered)/len(actual) if actual else 0,
            "answered_accuracy": sum(actual[i] == predicted[i] for i in answered)/len(answered) if answered else None,
            "overall_accuracy": correct/len(actual) if actual else 0,
            "answered": len(answered), "correct": correct, "total": len(actual)}
