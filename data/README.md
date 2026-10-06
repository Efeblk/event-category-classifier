# Dataset

`python prepare_events.py` downloads the source files and creates `events.csv` and `events.audit.json` locally. They are not committed.

## Source

- [rebrowser/gametime-dataset](https://huggingface.co/datasets/rebrowser/gametime-dataset) on Hugging Face, also on [Kaggle](https://www.kaggle.com/datasets/rebrowser/gametime-dataset). Event listings from the Gametime ticket marketplace, published by Rebrowser.
- License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial course use with attribution.
- Pinned commit `f24a84e1bedee99bef273332fe705cc7ed1958be`: 30 daily `events/data/*.csv` files, 2026-08-24 to 2026-09-22.

## Preparation

- Category mapping: `music` → `concert`, `theater` → `theatre`, `comedy` → `stand_up`. Sports and other categories are dropped.
- Text is the event title. Ticketing notes such as "(21+ Event)", "(Rescheduled from 3/28)", and "(Open Caption)" are removed.
- 76 Toronto International Film Festival screenings are dropped. Gametime lists them as theater, but they are films.
- Repeated dates of one show become one row. 15 titles that appear under two categories are dropped.
- `group_id` is the first listed performer, so one performer's titles stay in one partition.

Result: **3,506 rows**, 2,831 groups. 2,575 concert, 564 stand-up, 367 theatre.

CSV fields are `id,text,label,group_id,title,source,source_url`. `title` is the original title. `source_url` points to the pinned source file.

## Known limits

- English and US-centric.
- Many titles are only a name ("Metallica", "Kevin Hart"). The category then depends on knowing the performer.
- Labels come from the ticket site and were not reviewed by hand. Gametime's theater category is broad: plays, musicals, ballet, opera, circus, and some orchestra concerts.
- The classes are imbalanced: 73% concert.

## Request challenge set

`request_challenge.json` has 40 AI-authored Turkish and English requests (CC0, see [LICENSE](LICENSE)). It was written for an earlier version of this project and inspected during development. Here it is a secondary transfer test, not a blind benchmark. It includes an `unclear` label, which the title models cannot predict except when no feature is known.
