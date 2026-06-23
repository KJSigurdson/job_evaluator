# EA Job Evaluator

Daily pipeline that scrapes EA-aligned job boards, scores new roles against a personal profile, and inserts high-fit roles into a Notion tracker with LLM-generated reasoning and CV guidance.

## Design decisions and known limitations

**Insert-only Notion writes.** The pipeline only ever inserts new rows. If a role's URL already exists in Notion, the run skips it entirely — no field overwrite, no status reset. Once you move a role to Draft/Applied, a later run re-encountering that posting leaves it untouched.

**Source retrieval is capped at the 1000 most recent postings per board (Algolia search-tier limit); older roles beyond that are not ingested. Acceptable because the pipeline targets newly-posted roles and dedupes against history.**
