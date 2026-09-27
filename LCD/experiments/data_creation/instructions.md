# Data-creation instructions

This is the protocol for the planned local Python workflow; there is no runnable data-creation CLI yet. Expert-selected articles and contradiction-modification rules are required inputs. Do not substitute agent-chosen articles or rules.

## Collection and expert review

1. Human experts provide the target legal articles. Subagents find original Russian court rulings concerning those articles from accessible internet sources. Record a stable source URL, retrieval date, document identifier, and full text or a local copy for expert review.
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
