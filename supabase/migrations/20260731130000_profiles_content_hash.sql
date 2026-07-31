-- True North v2: re-embed-on-change support for profiles.profile_embedding.
-- src/profile_embed.py hashes the rendered embedding-input text (job_normalize.
-- content_hash — sha256 of lowercased/whitespace-collapsed text) and compares it
-- against this column before calling Voyage, so an unchanged profile is never
-- re-embedded. Additive, nullable, no default.
alter table profiles add column if not exists profile_content_hash text;
