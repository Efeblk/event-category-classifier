# Data licences

The code in this repository is MIT licensed (see `LICENSE`). The files under `data/` keep their own licences, as
described below.

## Schema-Guided Dialogue (SGD): `data/raw/sgd/` and `data/clean/{train,dev,test}.jsonl`

- Source: Google Research, Schema-Guided Dialogue dataset,
  https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
- Licence: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0),
  https://creativecommons.org/licenses/by-sa/4.0/. The full licence text is in `data/raw/sgd/LICENSE.txt`.
- Modifications: `prepare_sgd.py` keeps only dialogues with an `Events*` service and only the USER turns with Events
  frames. It keeps the Events slots (`city_of_event` is renamed to `city`), drops the category, subcategory and
  event_type slots, and turns the spans into BIO tags over word tokens. It then removes duplicate texts and texts that
  also appear in an earlier split. The output is a subset with new tags and no dialogue context.
- Because the data is shared under BY-SA, derived files must be shared under the same licence.

## Amazon MASSIVE: `data/raw/massive/` and `data/clean/massive_{train,dev,test}.jsonl`

- Source: Amazon MASSIVE en-US, https://github.com/alexa/massive
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/. The licence file is in `data/raw/massive/LICENSE`.
- Citation: FitzGerald, J., Hench, C., Peris, C., et al. (2022). MASSIVE: A 1M-Example Multilingual Natural Language
  Understanding Dataset with 51 Typologically-Diverse Languages. arXiv:2204.08582.
- Modifications: `prepare_massive.py` keeps the en-US rows that have a date, place_name or event_name slot. It maps
  place_name to city, keeps event_name only for the recommendation_events intent, converts the annotations to BIO tags
  over word tokens, and removes duplicate texts and texts that also appear in an earlier split.

## Messy test set: `data/messy/messy_test.jsonl`

- Written by the author: AI-drafted requests that were checked and corrected by hand. Not taken from any other source.
- Licence: MIT, the same as the code.

## Pre-trained models (not redistributed)

`train_bert.py` downloads these from Hugging Face when it runs. They are not included in this repository.

- `bert-base-uncased`: Apache-2.0, https://huggingface.co/bert-base-uncased
- `roberta-base`: MIT, https://huggingface.co/roberta-base
- `vinai/bertweet-base`: MIT, https://huggingface.co/vinai/bertweet-base

The licences above were checked against the model cards' `license:` fields on Hugging Face.
