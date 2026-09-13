# EAAI submission package

This directory tracks the files and decisions needed for submission to
*Engineering Applications of Artificial Intelligence* (EAAI). The journal uses
double-anonymized review and requires a single-column manuscript.

## Files prepared

- `../main.tex`: anonymous, single-column Elsevier manuscript.
- `../main.pdf`: anonymous review PDF produced from `main.tex`.
- `../title_page.tex` and `../title_page.pdf`: separate author title page.
- `../supplementary_material.tex` and `../supplementary_material.pdf`:
  anonymous detailed results and implementation notes, separated to keep the
  manuscript below the journal's 50-page limit.
- `highlights.txt`: five submission highlights.
- `cover_letter.md` and `cover_letter.docx`: editable cover-letter draft.
- `declaration_of_competing_interests.md` and `.docx`: confirmed no-conflict
  declaration.
- `author_contributions.md` and `.docx`: CRediT worksheet.
- `author_approval_record.md` and `.docx`: email/signature record for final
  authorship and submission approval.
- `data_availability_options.md`: two ready-to-use statements depending on
  whether a public project repository will be supplied.
- `submission_metadata.md`: title, authors, affiliations, keywords, and fields
  that must be copied into Editorial Manager.
- `artwork_manifest.md`: figure-to-source-file map and resolution audit.
- `submission_checklist.md`: final pre-submission quality gate.

## Blocking items

Do not submit until every `[REQUIRED]` item in `submission_checklist.md` is
resolved. The current blockers are final CRediT roles, author approval, and a
data-availability decision for the study-specific sample manifests, derived
annotations, and evaluation code. Artwork provenance and the proposed Inspec
codes must also be confirmed during submission.

## Suggested upload order

1. Anonymous manuscript source package and `main.pdf`.
2. Separate `title_page.pdf` and editable `title_page.tex`.
3. `highlights.txt`.
4. Final declaration-of-interests Word file generated or confirmed through
   Elsevier's declarations workflow.
5. Each figure as a separately named artwork file.
6. `supplementary_material.pdf` and its editable LaTeX source package.
7. Cover letter.

Before upload, remove this README, all working notes, and every file containing
`[REQUIRED]`, `TODO`, or `TO BE CONFIRMED` from the source package sent to the
journal.
