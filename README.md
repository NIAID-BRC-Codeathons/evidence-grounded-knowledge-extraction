# Evidence-Grounded Pathogen Knowledge Extraction

**NIAID-BRCs AI Codeathon 2.0** · September 16–18, 2026 · Argonne National Laboratory

A reusable literature-curation service that extracts pathogen-related claims and maps them to BRC entities and schemas with passage-level provenance.

Project page: https://niaid-brc-codeathons.github.io/projects/evidence-grounded-knowledge-extraction/

---

> **This is a draft pitch, not a plan.**
>
> What follows is a one-slide proposal from the organizing team. It exists
> to seed a team, not to constrain one. Scope, methods, target organism,
> and success criteria are all still open — expect them to change
> substantially. Turning this into a real plan is the team's first job, and
> it lands in the project charter due August 28, 2026.

---

## Goal (proposed)

Develop a reusable literature-curation service that extracts pathogen-related claims and maps them to BRC entities and schemas with passage-level provenance.

## Three-Day MVP (proposed)

Process 200–500 open-access papers and extract three relation types, such as pathogen–host phenotype, pathogen–mechanism–disease, and biomarker–disease. Produce structured JSON-LD or RDF, BV-BRC/NCBI identifiers, confidence scores, and a “cite or refuse” validation step. COMIC/InfectioVision and glycan-biomarker extraction can serve as the first two plug-in use cases.

## Model and evaluation (proposed)

Calibrate or fine-tune an extraction model on a small manually reviewed training set. Evaluate entity linking, relation precision/recall, citation correctness, and unsupported-claim rate. Incorporate GDB-Lit-style benchmark tasks.

## Leads

- Vijayaraj Nagarajan
- Maulik Shukla

Team assignments are still being finalized. Participants can review their project, and request a reassignment, in the participant spreadsheet circulated by the organizing team.

## Working here

This repository is the team's working space for the codeathon — code, notebooks, data pointers, and notes. Replace this README with the real thing once the charter is written. Team members get access through the [NIAID-BRC-Codeathons](https://github.com/NIAID-BRC-Codeathons) organization; accept the invitation if you have not already.
