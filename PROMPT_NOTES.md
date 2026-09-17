# Prompt Engineering Notes

Log of prompt iterations and why each change was made. Useful both as a build diary and
as interview talking points.

## v1 — Naive prompt
```
Answer this question using the following context: {context}
Question: {query}
```
**Problem:** Model answered confidently even when context was irrelevant — hallucinated
plausible-sounding but wrong answers when retrieval missed.

## v2 — Added explicit grounding instruction
Added: *"If the context does not contain enough information, say so explicitly."*
**Result:** Model started refusing appropriately on out-of-scope questions, but refusals
were vague ("I don't have enough information") without saying what it *did* find.

## v3 (current) — Source attribution + concise constraint
Added: *"Cite which source document each part of your answer comes from"* and
*"Keep answers concise and direct."*
**Result:** Answers now show provenance per claim, which (a) builds user trust and
(b) makes it easy to spot retrieval failures — if the citation looks unrelated to the
question, that's a signal the retriever, not the generator, needs work.

## Open experiments / next steps
- Try few-shot examples of "good refusal" vs "good answer" to sharpen the boundary.
- Test structured JSON output (answer + sources + confidence) for easier UI parsing.
- Add a query-rewriting step for vague questions before retrieval (query expansion).
