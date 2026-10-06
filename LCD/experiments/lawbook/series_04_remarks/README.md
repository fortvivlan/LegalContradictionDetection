# Series 04: КоАП remarks and application scopes

This local workflow extracts legal `Примечание` / `Примечания` text from the supplied КоАП DOCX and annotates its immediate application scope for expert review and later lawbook linking. Each numbered remark gets its own row; continuation paragraphs and internal lists stay together. Publisher commentary and amendment notices are excluded using DOCX styles. Repealed placeholders retain their source locations and receive no active application links.

The taxonomy distinguishes code-wide definitions, individual articles, elements of an article, multiple articles, chapters/sections, mixed scopes, other remarks, external norms, thematic groups requiring clarification, and repealed text. Application targets are separate from citations defining conditions, exceptions, or borrowed definitions. Part ranges become individual structured entities, and references to other remarks retain their own entity type. Edition and text hashes protect curated annotations from stale reuse.

Artifacts are `koap_remarks.csv` (one row per remark with full text, category, target JSON and provenance), `koap_remarks_review.xlsx` (editable expert review with text/category/entities first), `remark_categories.txt` (Russian definitions and linking conventions), and `remark_targets.csv` (one immediate target link per row). `scope_annotations.json`, `extracted_remarks.json`, and `manifest.json` preserve the provisional annotation and extraction record under ignored `local/`.

The first supplied-document export contains 281 rows across 173 remark blocks, including 11 repealed placeholders. All annotations await expert review. Context-inferred parts and open thematic scopes receive priority review; suggested article candidates are separate from application links. This experiment prepares annotations and does not alter the retrieval corpus or compute transitive application through other remarks.

Run and review guidance is in the [lawbook instructions](../instructions.md#series-04-koap-remarks-and-application-entities).
