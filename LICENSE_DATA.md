# License: `manifest_public.csv`

`manifest_public.csv` (and its checksum cache, `source_audio_checksums.csv`) is licensed under
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

Full legal text: https://creativecommons.org/licenses/by-nc/4.0/legalcode
Human-readable summary: https://creativecommons.org/licenses/by-nc/4.0/

## Why CC BY-NC, not a fully permissive license

This file contains verbatim quotations of direct speech (`speech` column) extracted from
third-party copyrighted literary works, and links to third-party copyrighted audio recordings
(`knigavuhe_link`) — we do not hold rights to the underlying works themselves, only to the
metadata and annotations we produced (LLM-extracted `description`, acoustic/manner scores,
train/val/test split, etc.). Non-commercial, attribution-required redistribution is the more
defensible choice given this: it keeps use of the file within research/educational purposes
(consistent with how it is intended to be used — reproducing the training set, per `README.md`)
without granting rights to the copyrighted excerpts it references beyond what the metadata itself
requires.

This license applies **only** to `manifest_public.csv`/`source_audio_checksums.csv` (the dataset
metadata). The pipeline code (`scripts/`, `build_public_manifest.py`) is licensed separately under
MIT — see [`LICENSE`](LICENSE). See the "License" section of [`README.md`](README.md) for the
summary of both.
