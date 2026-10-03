"""Create the small synthetic request-intent prototype corpus.

The examples are authored for this repository. They are not observed user data.
They are sufficient for a local prototype, but they do not satisfy a 1,000-example
course requirement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


LABELS = ("concert", "theatre", "stand_up", "unclear")
COLUMNS = ("id", "text", "label", "group_id", "title", "source", "source_url")
SOURCE = "generated-bootstrap"

# Each tuple is one semantic family. Keep a family in one data partition.
FAMILIES: dict[str, tuple[tuple[str, ...], ...]] = {
    "concert": (
        ("Canlı müzik dinlemek istiyorum", "Bu akşam bir konsere gidelim", "Bana canlı müzik etkinliği öner", "Sahnede müzik olan bir etkinlik arıyorum"),
        ("İyi bir rock konseri bul", "Canlı rock grubu izlemek istiyorum", "Bu hafta sonu rock müzik etkinliği var mı", "Elektro gitar ağırlıklı bir konser öner"),
        ("Pop konserine gitmek istiyorum", "Sevilen bir popçuyu canlı dinleyelim", "Dans edebileceğim bir pop konseri öner", "Yakındaki pop müzik sahnelerini göster"),
        ("Caz konseri arıyorum", "Canlı caz dinleyebileceğim bir gece öner", "Bir caz grubu sahnede olsun", "Saksafonlu bir caz performansına gitmek istiyorum"),
        ("Klasik müzik konseri öner", "Bir senfoni orkestrasını canlı dinlemek istiyorum", "Oda müziği konseri bulabilir misin", "Piyano resitali için öneri ver"),
        ("Rap konserine gitmek istiyorum", "Canlı hip hop performansı bul", "Sahnede rapçi olan bir etkinlik öner", "Bu hafta bir rap konseri var mı"),
        ("Elektronik müzik konseri arıyorum", "Canlı elektronik müzik performansı öner", "Sahnede elektronik müzik dinlemek istiyorum", "Bir elektronik müzik gecesine bilet bakıyorum"),
        ("Bağımsız müzik gruplarının konserlerini göster", "Bir indie grubunu canlı izlemek istiyorum", "Alternatif müzik konseri öner", "Yeni grupların çaldığı bir konser bul"),
        ("Türk halk müziği konseri istiyorum", "Canlı türkü dinleyebileceğim bir etkinlik bul", "Bağlama çalınan bir konser öner", "Halk müziği sanatçısını sahnede izlemek istiyorum"),
        ("Akustik konser öner", "Sakin bir canlı akustik performans arıyorum", "Gitar ve vokal ağırlıklı konser bul", "Küçük sahnede akustik müzik dinlemek istiyorum"),
        ("Sevdiğim şarkıcıyı sahnede görmek istiyorum", "Bir müzik grubunun canlı performansına gidelim", "Sanatçıların konser takviminden bir etkinlik öner", "Ünlü bir solisti canlı dinleyebileceğim konser bul"),
        ("Tiyatro değil, konser istiyorum", "Stand up olmasın; canlı müzik öner", "Oyun izlemek istemiyorum, bir konsere gidelim", "Komedi gösterisi yerine müzik konseri bul"),
        ("konser onerir misin", "canli muzik etkinligi bul", "bi rok konserine gidelim", "muzisyenleri sahnede izlemek istiyom"),
    ),
    "theatre": (
        ("Bir tiyatro oyunu izlemek istiyorum", "Bu akşam sahnelenen bir oyun öner", "Bana güzel bir tiyatro etkinliği bul", "Oyuncuları sahnede izleyebileceğim bir oyun arıyorum"),
        ("Dram türünde bir tiyatro oyunu öner", "Duygusal bir sahne oyunu izlemek istiyorum", "Ciddi bir tiyatro draması bul", "Dramatik bir oyun için bilet bakıyorum"),
        ("Shakespeare oyunu izlemek istiyorum", "Klasik bir tiyatro eseri öner", "Eski bir metnin sahne uyarlamasını bul", "Klasiklerden sahnelenen bir oyun var mı"),
        ("Çocuklar için tiyatro oyunu öner", "Ailece izleyebileceğimiz bir çocuk oyunu bul", "Çocuğumu bir tiyatro oyununa götürmek istiyorum", "Küçük izleyicilere uygun sahne oyunu arıyorum"),
        ("Sahnede oynanan bir müzikal izlemek istiyorum", "Tiyatro müzikali için bilet bul", "Oyunculu ve şarkılı bir sahne oyunu öner", "Bir Broadway tarzı tiyatro müzikali arıyorum"),
        ("Tek kişilik tiyatro oyunu öner", "Sahnede bir oyuncunun olduğu monolog izlemek istiyorum", "Bir monodram bulabilir misin", "Tek oyunculu bir tiyatro gösterisine gidelim"),
        ("Yeni yazılmış çağdaş bir oyun öner", "Modern tiyatro yapımı izlemek istiyorum", "Güncel bir metnin sahnelendiği oyun bul", "Çağdaş tiyatro topluluklarının oyunlarını göster"),
        ("Absürt tiyatro oyunu arıyorum", "Gerçeküstü bir sahne oyunu öner", "Absürt bir metnin tiyatro uyarlamasını izleyelim", "Deneysel ve absürt bir tiyatro yapımı bul"),
        ("Tarih anlatan bir tiyatro oyunu öner", "Dönem kostümlü bir sahne oyunu izlemek istiyorum", "Tarihî bir olayın tiyatro uyarlamasını bul", "Geçmişte geçen bir oyun için bilet arıyorum"),
        ("Kukla tiyatrosu izlemek istiyoruz", "Çocuklar için kukla oyunu bul", "Sahnede kuklaların olduğu bir tiyatro öner", "Bir kukla gösterisine bilet bakıyorum"),
        ("Roman uyarlaması bir tiyatro oyunu öner", "Bir kitabın sahneye uyarlandığı oyun bul", "Edebî eser uyarlaması izlemek istiyorum", "Hikâyeden uyarlanmış bir sahne oyunu arıyorum"),
        ("Konser değil, tiyatro oyunu istiyorum", "Stand up olmasın; sahne oyunu öner", "Canlı müzik yerine bir tiyatro yapımı bul", "Komedyen gösterisi istemiyorum, oyunculu bir oyun olsun"),
        ("tiyatro oyunu oner", "sahne oyunu izlemek istiyom", "bi tiyatro bileti bulur musun", "oyuncularin oynadigi bi oyun ariyorum"),
    ),
    "stand_up": (
        ("Stand-up gösterisine gitmek istiyorum", "Bana bir stand up gecesi öner", "Sahnede komedyen izlemek istiyorum", "Canlı stand-up etkinliği bul"),
        ("Sevilen bir komedyenin gösterisini bul", "Bir komedyeni sahnede izleyelim", "Tek kişilik komedyen gösterisi öner", "Canlı komedyen performansına bilet arıyorum"),
        ("Stand-up açık mikrofon gecesi öner", "Yeni komedyenlerin çıktığı open mic bul", "Açık mikrofonda stand up izlemek istiyorum", "Amatör komedyenlerin sahne aldığı geceyi göster"),
        ("Tek kişilik stand-up gösterisi arıyorum", "Bir stand-up sanatçısının solo şovunu öner", "Sahnede yalnız bir komedyenin olduğu etkinlik bul", "Solo komedi performansına gitmek istiyorum"),
        ("Siyasi taşlama yapan stand-up gösterisi bul", "Güncel olayları tiye alan bir komedyen izlemek istiyorum", "Siyasi mizah içeren stand up öner", "Toplumsal hiciv yapan komedyenin gösterisini bul"),
        ("Gündelik hayat üzerine stand-up öner", "İlişkileri anlatan bir komedyen izlemek istiyorum", "Gözlemsel mizah yapan stand-upçı bul", "Günlük dertlerle dalga geçen bir stand up gösterisi arıyorum"),
        ("Genç bir stand-up komedyeni öner", "Yeni nesil komedyenlerin gösterilerini bul", "Yeni yüzlerin stand up gecesine gitmek istiyorum", "Kariyerinin başındaki bir komedyeni sahnede izleyelim"),
        ("İngilizce stand-up gösterisi bul", "İngilizce konuşan bir komedyeni canlı izlemek istiyorum", "English stand up night öner", "Yabancı dilde bir stand-up etkinliği arıyorum"),
        ("Kadın komedyenlerin stand-up gösterilerini öner", "Sahnede kadın bir stand-upçı izlemek istiyorum", "Kadın komedyen gecesi bul", "Bir kadın komedyenin solo gösterisine gidelim"),
        ("Roast tarzı bir stand-up gecesi bul", "Sahnede laf sokmalı komedyen gösterisi öner", "Roast yapan komedyenleri canlı izlemek istiyorum", "Sert şakalı bir stand up etkinliği arıyorum"),
        ("Hikâye anlatan bir stand-upçı izlemek istiyorum", "Uzun hikâyeli stand-up gösterisi öner", "Anılarını komik biçimde anlatan komedyen bul", "Öykülü bir stand up performansına gidelim"),
        ("Tiyatro değil, stand-up istiyorum", "Konser olmasın; komedyen gösterisi öner", "Sahne oyunu yerine stand up bul", "Canlı müzik istemiyorum, komedyen izleyelim"),
        ("standap gosterisi oner", "sitendap izlemek istiyom", "bi komedyen sovu bul", "stand up bileti bakar misin"),
    ),
    "unclear": (
        ("Bir şeyler yapmak istiyorum", "Bana bir etkinlik öner", "Bu akşam dışarı çıkalım", "Hafta sonu için plan yapalım"),
        ("İyi bir restoran öner", "Yakında kahve içilecek yer bul", "Akşam yemeği için mekân arıyorum", "Güzel bir bar biliyor musun"),
        ("Bütçem beş yüz lira", "Kişi başı üç yüz lirayı geçmesin", "Ucuz bir şey olsun", "Toplam bin liramız var"),
        ("Cuma akşamı müsaitim", "Yarın saat sekizden sonra", "Ayın on beşinde olsun", "Bu hafta sonu gündüz vakti"),
        ("Kadıköy tarafında olsun", "Avrupa yakasında bir yer arıyorum", "Eve yakın olsun", "Beşiktaş çevresinde ne var"),
        ("Biraz eğlenmek istiyoruz", "Keyfimizi yerine getirecek bir şey öner", "Neşeli bir akşam olsun", "Kafamı dağıtacak bir plan arıyorum"),
        ("Konser ya da tiyatro olabilir", "Canlı müzikle sahne oyunu arasında kararsızım", "Tiyatro da konser de uyar", "Ya bir oyun ya da bir konser öner"),
        ("Konser istemiyorum", "Tiyatro olmasın", "Stand up görmek istemem", "Canlı müzikten hoşlanmıyorum"),
        ("Biraz müzik aç", "Bana bir şarkı çal", "Çalma listemi başlat", "Kulaklıkta caz çalabilir misin"),
        ("Sinemada film izlemek istiyorum", "Yeni çıkan bir filme bilet bul", "Bu akşam hangi film var", "Bir sinema gösterimi öner"),
        ("Futbol maçı izlemek istiyorum", "Bir sergi öner", "Müzeye gitmek istiyorum", "Basketbol karşılaşması bul"),
        ("Gülelim biraz", "Komik bir etkinlik olsun", "Mizah içeren bir şey öner", "Eğlenceli bir gösteri arıyorum"),
        ("Biletler kaldı mı", "Etkinlik kaçta bitiyor", "İade koşulları nedir", "Mekâna nasıl giderim"),
    ),
}

# These independently phrased English requests match the semantic families above.
# A translated intent uses the same group_id in both languages.
ENGLISH_FAMILIES: dict[str, tuple[tuple[str, ...], ...]] = {
    "concert": (
        ("I want to hear live music", "Find me a concert for tonight", "Suggest an event with musicians performing live"),
        ("Find a good rock concert", "I want to see a rock band live", "Suggest a concert with plenty of electric guitar"),
        ("I would like to go to a pop concert", "Find a live performance by a pop singer", "Suggest a pop show where we can dance"),
        ("I am looking for a jazz concert", "Suggest a night with live jazz", "Find a jazz band playing on stage"),
        ("Suggest a classical music concert", "I want to hear a symphony orchestra live", "Find a piano recital or chamber concert"),
        ("I want to go to a rap concert", "Find a live hip hop performance", "Suggest an event where a rapper is on stage"),
        ("I am looking for an electronic music concert", "Suggest a live electronic music performance", "Find tickets for an electronic music night"),
        ("Show me concerts by independent bands", "I want to see an indie band live", "Suggest an alternative music concert"),
        ("I want a Turkish folk music concert", "Find an event with live folk songs", "Suggest a concert featuring the bağlama"),
        ("Suggest an acoustic concert", "I want a quiet live acoustic performance", "Find a small concert built around guitar and vocals"),
        ("I want to see my favorite singer on stage", "Let us watch a band perform live", "Find a concert by a well-known vocalist"),
        ("I want a concert, not a play", "No stand-up please, suggest live music", "Find a music concert instead of a comedy show"),
        ("can u suggest a consert", "find me some live musik", "wanna see a rok band live"),
    ),
    "theatre": (
        ("I want to watch a theatre play", "Suggest a play being staged tonight", "Find an event with actors performing on stage"),
        ("Suggest a dramatic theatre play", "I want to watch an emotional stage play", "Find tickets for a serious drama"),
        ("I want to watch a Shakespeare play", "Suggest a classic work of theatre", "Find a stage adaptation of an old text"),
        ("Suggest a theatre play for children", "Find a children's play for the family", "I want to take my child to a stage play"),
        ("I want to watch a musical performed on stage", "Find tickets for a theatre musical", "Suggest a stage play with acting and songs"),
        ("Suggest a one-person theatre play", "I want to watch an actor perform a monologue", "Find a solo stage drama"),
        ("Suggest a newly written contemporary play", "I want to see a modern theatre production", "Find a play by a contemporary theatre company"),
        ("I am looking for absurdist theatre", "Suggest a surreal stage play", "Find an experimental absurdist theatre production"),
        ("Suggest a theatre play about history", "I want a stage play with period costumes", "Find a play based on a historical event"),
        ("We want to watch puppet theatre", "Find a puppet play for children", "Suggest a theatre show performed with puppets"),
        ("Suggest a theatre play adapted from a novel", "Find a book adaptation on stage", "I want to watch a literary work turned into a play"),
        ("I want a theatre play, not a concert", "No stand-up please, suggest a stage play", "Find an acted play instead of live music"),
        ("suggest a theter play", "wanna see a stage ply", "find me a ticket for a teatr show"),
    ),
    "stand_up": (
        ("I want to go to a stand-up show", "Suggest a stand up night", "Find a live stand-up event"),
        ("Find a show by a popular comedian", "Let us watch a comedian on stage", "Suggest a live comedian performance"),
        ("Suggest a stand-up open mic night", "Find an open mic with new comedians", "I want to watch amateur comics perform"),
        ("I am looking for a solo stand-up show", "Suggest one comedian's solo show", "Find an event with one comic on stage"),
        ("Find political satire stand-up", "I want to watch a comedian joke about current events", "Suggest a stand-up show with political humor"),
        ("Suggest stand-up about everyday life", "I want a comedian who jokes about relationships", "Find an observational comedy show"),
        ("Suggest a young stand-up comedian", "Find shows by a new generation of comics", "I want to see an emerging comedian live"),
        ("Find an English-language stand-up show", "I want to watch a comedian perform in English", "Suggest an English stand up night"),
        ("Suggest stand-up shows by women comedians", "I want to see a woman comic on stage", "Find a solo show by a woman comedian"),
        ("Find a roast-style stand-up night", "Suggest a comedian show with sharp insults", "I want to watch comics perform a roast"),
        ("I want a stand-up comedian who tells stories", "Suggest a stand-up show with long stories", "Find a comic who makes funny stories from personal memories"),
        ("I want stand-up, not theatre", "No concert please, suggest a comedian show", "Find stand up instead of a stage play"),
        ("suggest a standap show", "wanna see a comedien live", "find me a stand up tcket"),
    ),
    "unclear": (
        ("I want to do something", "Suggest an event for me", "Let us make a plan for the weekend"),
        ("Suggest a good restaurant", "Find a nearby coffee shop", "I am looking for a place to eat dinner"),
        ("My budget is five hundred lira", "Keep it under three hundred per person", "We have one thousand lira in total"),
        ("I am free on Friday evening", "Tomorrow after eight works", "Make it sometime this weekend"),
        ("It should be around Kadıköy", "Find something on the European side", "What is available near Beşiktaş"),
        ("We want to have some fun", "Suggest something to cheer me up", "I need a fun plan for the evening"),
        ("It can be a concert or theatre", "I cannot choose between live music and a play", "Either a play or a concert works"),
        ("I do not want a concert", "No theatre for me", "I do not want to see stand-up"),
        ("Play some music", "Put on a song for me", "Start my jazz playlist"),
        ("I want to see a movie at the cinema", "Find tickets for a new film", "Suggest a movie screening tonight"),
        ("I want to watch a football match", "Suggest an art exhibition", "Find a basketball game"),
        ("Let us have a laugh", "Suggest something funny", "I am looking for an entertaining show"),
        ("Are there tickets left", "What time does the event finish", "How do I get to the venue"),
    ),
}


def normalize(text: str) -> str:
    """Return stable Turkish-aware text for duplicate checks and row IDs."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_rows() -> tuple[list[dict[str, str]], dict[str, object]]:
    """Build rows and reject duplicate text or conflicting normalized text."""
    rows: list[dict[str, str]] = []
    seen: dict[str, tuple[str, str]] = {}
    family_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()

    if set(FAMILIES) != set(LABELS):
        raise ValueError(f"Families must use exactly these labels: {', '.join(LABELS)}")
    if set(ENGLISH_FAMILIES) != set(LABELS):
        raise ValueError(f"English families must use exactly these labels: {', '.join(LABELS)}")

    for label in LABELS:
        if len(FAMILIES[label]) != len(ENGLISH_FAMILIES[label]):
            raise ValueError(f"Turkish and English family counts differ for {label}")
        paired_families = zip(FAMILIES[label], ENGLISH_FAMILIES[label], strict=True)
        for family_number, (turkish_examples, english_examples) in enumerate(paired_families, start=1):
            group_id = f"{label}-{family_number:02d}"
            if len(turkish_examples) < 3 or len(english_examples) < 3:
                raise ValueError(f"{group_id} needs at least three distinct utterances")
            family_counts[label] += 1
            examples = tuple((text, "Turkish") for text in turkish_examples) + tuple(
                (text, "English") for text in english_examples
            )
            for text, language in examples:
                normalized = normalize(text)
                if not normalized:
                    raise ValueError(f"{group_id} contains empty text")
                previous = seen.get(normalized)
                if previous:
                    old_label, old_group = previous
                    conflict = "label conflict" if old_label != label else "duplicate"
                    raise ValueError(
                        f"Normalized {conflict}: {text!r} in {group_id}; first in {old_group}"
                    )
                seen[normalized] = (label, group_id)
                language_counts[language] += 1
                rows.append(
                    {
                        "id": fingerprint(normalized),
                        "text": text,
                        "label": label,
                        "group_id": group_id,
                        "title": "",
                        "source": SOURCE,
                        "source_url": "",
                    }
                )

    rows.sort(key=lambda row: row["id"])
    class_counts = Counter(row["label"] for row in rows)
    audit: dict[str, object] = {
        "unique_examples": len(rows),
        "intent_families": sum(family_counts.values()),
        "class_counts": dict(sorted(class_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "labels": list(LABELS),
        "languages": ["Turkish", "English"],
        "provenance": "AI-authored synthetic prototype. No observed real-user requests.",
        "license": "CC0-1.0",
        "third_party_text": False,
        "course_requirement_status": "Does not satisfy the 1,000-example requirement.",
        "group_rule": "Paraphrases of one base intent share a stable class-NN group_id.",
    }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/requests.csv"))
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Fail below this row count. The prototype default is 1.",
    )
    args = parser.parse_args()
    if args.min_rows < 1:
        parser.error("--min-rows must be positive")

    rows, audit = build_rows()
    if len(rows) < args.min_rows:
        parser.error(f"Need at least {args.min_rows} rows. Found {len(rows)}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    audit["output"] = str(args.output)
    audit["dataset_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
