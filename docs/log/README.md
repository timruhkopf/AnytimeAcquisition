# Research Log

A chronological, append-only record of things tried during this project —
distinct from `docs/ROADMAP.md` and `docs/milestones/M*.md`, which are the
current, living state of the design. This log never gets rewritten to match
the present; it's the history of how the present got decided, including
dead ends. When a log entry changes a design decision, update the relevant
roadmap/milestone file to reflect the new state and just link to it from
here — don't let this log become a second source of truth for "what's true
now."

## Adding an entry

Copy `TEMPLATE.md` to a new file named:

```
YYYY-MM-DD-descriptive-kebab-case-slug.md
```

The date keeps the directory sortable in a plain `ls`; the slug should be
specific enough to identify the idea from the filename alone in a `grep` or
directory listing, without opening the file. Every entry **must** include a
"What we learned" section, even (especially) when the idea failed or was
inconclusive — that section is the entire point of keeping this log.

Add a one-line pointer to the index below when you add an entry — newest
first.

## Index

<!-- newest first — add new entries above this line -->
- [2026-08-28 — Exploit-search targets may be more privileged than the state can justify](2026-08-28-exploit-search-target-may-outrun-context.md)
- [2026-08-28 — Explore-branch search: from a self-referential entropy objective to teacher-forced, privileged-NLL gradient descent](2026-08-28-explore-search-input-optimization-and-teacher-forcing.md)
- [2026-08-28 — Why pi0/pi0.5 use separate expert weights + a fine-tuned backbone, and why the ActionHead deliberately differs (frozen PFN)](2026-08-28-pi0-moe-and-frozen-vs-finetuned-backbone.md)
- [2026-08-28 — M2: PFN transformer + bar distribution, trained checkpoints, prior/data/PFN overlay notebook](2026-08-28-m2-pfn-and-bar-distribution.md)
- [2026-08-28 — Align BNNPrior with PFNs4BO/ifBO: noise, input scaling, sparseness, spurious dims, deeper default](2026-08-28-align-bnn-prior-with-pfns4bo-ifbo.md)
- [2026-08-27 — PFNs4BO's own BNN prior — what they do differently from ours](2026-08-27-pfns4bo-bnn-prior-comparison.md)
- [2026-08-27 — Most BNNPrior draws were flat — independent init_std/width sampling, fixed via crit-scaling](2026-08-27-bnn-prior-flat-draws-crit-scaling.md)
