# Data-creation instructions

Expert-selected articles are now available for document collection. Contradiction-modification rules remain an expert input for later dataset construction. Do not substitute agent-chosen articles or rules.

## Collection scope and sources

- Main article group: `18.8`, `18.9`, `18.10`, `18.11`, `18.12`, `18.15`, `18.16`, `18.17`, `18.18`, `18.19`, `18.20`, `19.27` КоАП РФ. Collect at least 1,000 distinct original judicial rulings across this group, seeking broad coverage and at least roughly ten per article where published decisions permit.
- Domain-shift group: `20.20`, `20.21`, `6.9` КоАП РФ. Collect about 30 original rulings, aiming at roughly ten per article. Keep this pool separate as test candidates; do not place its cases into train or dev.
- Collect final decisions on complaints separately for **both** groups, aiming for at least 30 verified appeals per article where published decisions permit and at least one for each article part where available. An appeal is evidence for expert review, not an original ruling or an automatically labeled contradiction.

Available public sources include [ГАС «Правосудие»](https://sudrf.ru/), which links to federal general-jurisdiction courts and magistrates; the [Moscow courts search](https://mos-gorsud.ru/mgs/search), which includes administrative-offence cases and review proceedings; [SudAct general courts](https://sudact.ru/regular/) and [SudAct magistrates](https://sudact.ru/magistrate/); and [ZakonRF's court-act archive](https://www.zakonrf.info/gorsud/). The current automated collectors use SudAct. Check official court portals and other public archives for coverage gaps and provenance confirmation where feasible. Record the exact document URL and source for every saved item. Search result titles, case-number prefixes, and page headings are discovery hints only; acceptance depends on the full published text. Use polite request intervals, bounded retries, and resume from saved provenance.

Store collected material under ignored `local/data/classification/dataset0929/decisions/` and `local/data/classification/dataset0929/appeals/`. Within each, keep `main/` and `domain_shift/` separate, then an article-number folder containing DOCX copies. Retain the original published HTML or text and a provenance manifest with retrieval time, source URL, article, court, case identifier, document type, content hash, and the passages supporting acceptance. Record inaccessible pages and shortages; never fill a target with duplicates, incomplete documents, or a different article.

Run the local collectors from the repository root with the project Python environment:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.collect_originals --group all \
  --count-main 1000 --count-domain-shift 30 --minimum-per-article 10
./.venv/bin/python -m LCD.experiments.data_creation.collect_appeals --group all \
  --target 30 --max-pages 8
./.venv/bin/python -m LCD.experiments.data_creation.audit_collection
```

Both commands are resumable against their provenance manifests. Use `--help` for search depth, timeouts, request intervals, year or article selection, and output paths. The DOCX files are text copies of published acts; the corresponding HTML and plain-text files preserve the fetched source. Inspect `provenance.csv` and `failures.csv` before expert review. A shortfall is a source-availability or retrieval result, not permission to include a document that fails the checks below.

Run appeal searches into a separate staging directory. Before copying any candidate into the active `appeals/` collection, inspect its published opening, charged original ruling, complaint, substantive reasoning, and final operative ruling. Confirm that the *same* article and part govern the appealed ruling; dates such as `18.11.2025`, case numbers such as `3/18.19-33/2025`, historical citations, and source typos do not establish the target article. Record per-document acceptance evidence or rejection reasons. Keep missing parts and per-article shortfalls visible in the review report.

## Collection and expert review

1. The original-rulings collector finds first-instance *постановления о привлечении к административной ответственности* under the selected articles. The published text must identify a КоАП РФ case, apply the target article in the final `ПОСТАНОВИЛ` section, and give an actual disposition such as guilt and penalty. Reject complaint reviews even when titled `Постановление`. The Chapter 30 complaint requirements below do not apply to original rulings. Record a stable source URL, retrieval date, document identifier, and full text or a local copy for expert review.
2. Subagents separately collect final court decisions on complaints against administrative-offence rulings under Chapter 30 of the КоАП РФ. The appeal pool supports expert analysis of errors and does not become the source-document corpus. Record any verified link to an original ruling, but matching is not required.
3. Subagents document the text passages supporting every appeal's inclusion. Human experts review the candidate rulings and appeals, and select corpus rulings before splitting them.

The expert appeal-search instructions are:

> **Задание для поиска.** Найти и собрать судебные акты РФ по делам об административных правонарушениях, в которых суд рассмотрел жалобу на постановление по делу об административном правонарушении в порядке главы 30 КоАП РФ. Нужны именно итоговые акты пересмотра постановлений: решения по жалобам на постановления, а не исходные постановления о привлечении к ответственности.
>
> Включать документы, в которых одновременно есть все признаки:
>
> 1. Прямо указано, что дело рассматривается по КоАП РФ / в порядке главы 30 КоАП РФ.
> 2. Есть обжалуемое постановление по делу об административном правонарушении: должностного лица, административного органа, комиссии или суда.
> 3. Суд рассматривает жалобу на это постановление.
> 4. В документе содержится итог рассмотрения жалобы: постановление оставлено без изменения, жалоба — без удовлетворения; постановление изменено; постановление отменено, производство прекращено; постановление отменено, дело возвращено на новое рассмотрение; либо постановление отменено, дело направлено по подведомственности.
>
> **Ключевое правило:** не доверять заголовку документа. Нужно по тексту подтвердить три вещи: есть постановление по КоАП, есть жалоба именно на него, и есть итоговое решение по этой жалобе.

## Split, transform, retrieve, and label

1. Divide accepted **original rulings**, not individual sentences or premise pairs, into `train`, `dev`, and `test`. Keep duplicate copies and related decisions from the same case in one split. Keep the appeal evidence pool separate. Record the split assignment and seed.
2. Extract hypotheses from the final `ПОСТАНОВИЛ` section using `razdel` sentence splitting and irrelevant-text filtering. Record the source sentence and any extraction or filtering exception. Experts approve changes to these rules before use.
3. Apply expert-provided modification algorithms to create contradictory sentence candidates in each split. Retain the original sentence, transformed sentence, rule identifier, and human review outcome. In `train`, seek one modified hypothesis for every two unmodified ones; if valid modifications are scarce, reduce unmodified hypotheses. Do not force `dev` or `test` to a one-third modification rate.
4. For **every** resulting hypothesis, run semantic lawbook retrieval and exact lookup for any explicit citation. Add exact-only results to semantic candidates; collapse the same provision returned by both routes into one pair, retaining both route names and source metadata. Keep unresolved citations visible for review.
5. Propose a three-class label for every candidate pair. Human experts approve all `dev` and `test` labels and synthetic modifications. For `train`, they review a stratified sample across articles, labels, and modification rules and every flagged or uncertain case. Preserve proposed and reviewed values separately.
6. Select eligible `train` pairs after the sample audit and flagged-case review toward 1:1:1; select expert-approved `dev` pairs toward approximately 60:20:20 (`not mentioned`: `entailment`: `contradiction`). Report the resulting counts and any shortage without inventing labels. Leave expert-approved `test` pairs unsampled, report their natural distribution, and note its distance from the hoped-for 70:25:5.

Future exports should preserve the existing classifier fields `premise`, `hypothesis`, `source`, and `tag`, alongside a provenance table linking each row to its document, URL, original sentence, modification rule, citation, retrieval routes, split, and review state. Put private inputs and outputs under ignored `local/` directories. The rebuilt splits are new inputs; historical `local/data/classification/val.csv` and benchmark `Full` remain separate.
