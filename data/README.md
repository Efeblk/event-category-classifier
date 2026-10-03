# Dataset

Keep `events.csv` and `events.audit.json` here. Git ignores both files.

The local dataset uses a frozen Bi’ Plan catalog. The text fields contain Turkish event titles and descriptions. The labels come from existing collector categories: `Konser`, `Tiyatro`, and `Stand-up`.

Run `prepare_data.py` with an existing JSON snapshot. The importer reads the file. It does not run a collector or change the source file.

The importer removes repeated text. It connects records with the same normalized title, exact text, long description, production key, or source page. It assigns one group ID to each connected family. The training program keeps each family in one partition. The importer excludes families with conflicting labels.

Provider labels can contain errors. Some event families can have different titles and descriptions. The grouping rules cannot detect all such cases. Review a label sample before submission. Do not treat this dataset as an independent human benchmark.

No explicit reuse license for provider text came with the snapshot. This repository does not grant a license for that text. Keep the corpus and models local until you confirm a suitable license or permission. The course also asks students to check the data license. A public repository can contain the code and aggregate results. It must not silently include the provider corpus.

To use another approved dataset, supply a CSV with these columns:

```text
id,text,label,group_id,title,source,source_url
```

Use the labels `concert`, `theatre`, and `stand_up`. Use at least 1,000 distinct examples. Use the same `group_id` for related or repeated events. Preserve the data source and license in this file.

## Source terms

Checked on 2026-10-03. These checks establish no open corpus license or permission for this project.

- [Biletinial policies](https://biletinial.com/tr-tr/sayfa/sozlesme-ve-politikalar), intellectual property section: the terms restrict content reuse and refer to written permission.
- [Bubilet terms](https://www.bubilet.com.tr/sayfa/kullanim-kosullari), content sections: the terms restrict use and distribution beyond stated personal uses.
- [Biletix terms](https://www.biletix.com/conditions/TURKIYE/en), clauses 1-4: the terms protect provider content and restrict copying and distribution.

Keep the existing corpus local. Confirm permission for course use and sharing. Use another suitable licensed dataset if permission is unavailable. Do not infer a data license from the fact that the source pages are public.
