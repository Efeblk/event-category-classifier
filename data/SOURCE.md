# Source, attribution and dataset license

**Dataset:** PTC-SemEval20, SemEval-2020 Task 11: Detection of Propaganda
Techniques in News Articles, `datasets-v2.tgz` (1,142,202 bytes).

**Creators:** Giovanni Da San Martino, Alberto Barrón-Cedeño, Henning Wachsmuth,
Rostislav Petrov and Preslav Nakov.

**Original release:** <https://zenodo.org/records/3952415>

**DOI:** <https://doi.org/10.5281/zenodo.3952415>

**Paper:** [SemEval-2020 Task 11: Detection of Propaganda Techniques in News
Articles](https://aclanthology.org/2020.semeval-1.186/), Proceedings of the
Fourteenth Workshop on Semantic Evaluation, 2020, pp. 1377–1414.

**Dataset license:** Creative Commons Attribution 4.0 International, as declared
by the [original record metadata](https://zenodo.org/api/records/3952415).
Full license text is in [LICENSE](LICENSE); source license URL:
<https://creativecommons.org/licenses/by/4.0/>. This license applies to the
distributed raw corpus and its derived cleaned examples. Repository source
code has its separate root license.

The 742 raw article/annotation files and source README are unmodified copies
from the verified original release. Changes to derived data are documented in
`README.md`, implemented in `../prepare_data.py`, and measured in
`cleaned/audit.json`: extract fragments/context, normalize Unicode whitespace,
remove one exact duplicate annotation and assign reproducible article groups.
The separate development files also reconstruct wider surrounding context from
only those groups assigned to train/dev, as documented in README.md and
cleaned/development_audit.json. Original benchmark files remain unchanged.
No annotation labels are invented, relabeled or translated.

The source article annotations can overlap and identical spans can have different
technique labels. We retain those labels as separate multiclass records, as in
the official technique-classification task. Official benchmark scoring matches
labels among identical spans without depending on row order. Our custom split
does not justify claiming a directly comparable official leaderboard score.

The paper describes 6,128 training annotations; the pinned archive actually has
6,129, including one exact duplicate. The cleaned table has 6,128 after the
documented duplicate removal. Counts are audited from source files rather than
silently copied from the paper.

```bibtex
@inproceedings{da-san-martino-etal-2020-semeval,
  title = "SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles",
  author = "Da San Martino, Giovanni and Barr{\'o}n-Cede{\~n}o, Alberto and Wachsmuth, Henning and Petrov, Rostislav and Nakov, Preslav",
  booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
  year = "2020",
  pages = "1377--1414",
  url = "https://aclanthology.org/2020.semeval-1.186/"
}
```
