# Architecture Decision Records

Decisions that shaped this service, each with the alternatives that were actually
considered, the evidence that settled it, and what would make it worth revisiting.

A record is written when a choice was **contested** — when a competent engineer could
reasonably have gone the other way, and the reason it went this way is not recoverable
from reading the code. Choices with one obvious answer are not recorded.

Records are immutable once accepted. A decision that changes gets a new record that
supersedes the old one, and the old one stays, because the reasoning that turned out to be
wrong is usually the more useful half.

| # | Decision | Status |
|---|---|---|
| [001](001-publish-the-model-losing.md) | Publish the fitted model losing to arithmetic on real data | Accepted |
| [002](002-alert-budget-over-precision-target.md) | Set the threshold from an alert budget, not a precision target | Accepted |
| [003](003-episode-metrics-over-point-accuracy.md) | Score episodes and alert volume, never accuracy | Accepted |

Portfolio-wide decisions live in
[`ai-engineering-portfolio/docs/adr/`](https://github.com/Adityansh-Chand/ai-engineering-portfolio/tree/main/docs/adr).
