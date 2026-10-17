# Proposal authoring

The authoritative proposal content is docs/proposal.md. The renderer reads that file directly, including its four page-break markers, and writes docs/proposal.pdf. Edit the Markdown once; do not maintain a second copy of proposal text in a script.

Using Python 3.12:

```powershell
python -m pip install -r requirements-docs.txt
python tools/render_proposal.py
```

ReportLab 4.4.9 was used successfully for the submitted artifact. The document-authoring dependency is separate from the application stack. The workspace-level build_proposal.py is only a compatibility wrapper around this renderer.

After edits, render every PDF page to PNG, inspect all pages for clipping/overlap/table wrapping, confirm four A4 pages and page numbers, and compare normalized extracted PDF text with the Markdown. Check word limits, team names/roll numbers and PDF author metadata. Keep the instructor's office-use fields empty. Do not publish stale render images or older drafts as current deliverables.
