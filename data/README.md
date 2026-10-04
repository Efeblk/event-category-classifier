# Dataset

`python prepare_requests.py` creates `requests.csv` locally.

- 364 AI-authored requests: 208 Turkish and 156 English.
- Four labels: `concert`, `theatre`, `stand_up`, `unclear`.
- 91 examples per label, in 52 paraphrase families.
- Translations and paraphrases share a `group_id` and stay in one partition.
- Original synthetic examples use [CC0-1.0](LICENSE). No observed user or ticket-provider text is included.

`unclear` means there is no single supported activity: vague requests, multiple acceptable categories, unsupported activities, or exclusions without a positive request.

The generator rejects duplicate normalized text and conflicting labels. CSV fields are `id,text,label,group_id,title,source,source_url`. The last three fields record source metadata; title and URL are empty for this synthetic dataset.

`request_challenge.json` has 40 extra AI-authored cases. These cases were inspected during development. They are regression cases, not a blind test.

The corpus is below the course's 1,000-example requirement. Add teacher-approved examples before submission. Keep related wording grouped and reserve unseen evaluation cases before changing the models.
