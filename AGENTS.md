# Agent instructions

## Project context

- Start with [README.md](README.md), then read the source documents relevant to the task.
- Follow the experiment design in the [proposal](CS3268_Project_Proposal_Group_6.md) exactly. Any deviation requires explicit user approval.
- Follow the [course requirements](docs/CS3268_Project_Document.md) and use the [cluster guide](docs/SoC_Cluster_Guide.md) for cluster work.

## Workflow and verification

- Follow the [contribution workflow](README.md#contributing) when submitting changes.
- Keep edits within the request. Inspect actual files before assuming scripts, commands, or results exist.
- Use subagents when they clearly save time or improve results. Give them clearly defined tasks that can be done independently, share relevant repository instructions, and avoid overlapping edits. The parent agent must review and combine their work, then check the final result. Prefer GPT-6.1 Sol or Opus 5.5 when available.
- Check affected work and results before drawing conclusions. Report checks and limitations; mark results that have not been checked. If bugs invalidate results, fix the bugs and rerun affected experiments when possible; otherwise provide rerun instructions. Keep affected conclusions pending until replacement results have been checked.

## Documentation and terminology

- Read [GLOSSARY.md](GLOSSARY.md) for preferred terms and definitions. Add or update definitions for necessary technical terms and abbreviations.
- Keep documentation, code comments, and other written explanations concise and easy to understand. Use plain language; use technical terms only when needed for precision.
- Use one preferred term for each concept across the repository. Follow the glossary's terms and abbreviations; avoid alternate names for the same concept.
- Give each documented fact one main source, including facts in code comments. Update that source; elsewhere, link with at most a one-sentence summary to avoid conflicting copies.

## Implementation style

This is a short course project. Favor the smallest clear implementation that meets the requirements and that teammates can explain to a professor or TA.

- Implement only current requirements. Do not design for future extensibility or add speculative features.
- Check the standard library first, then established libraries; reuse or write custom implementations only when neither suffices.
- Prefer plain functions and direct control flow. Add helpers, classes, or modules only when they simplify existing code or remove meaningful duplication. Avoid unnecessary wrappers and generic frameworks.
- Validate where mistakes could silently invalidate results or waste expensive runs. Avoid repeated checks; rely on library errors elsewhere. Catch exceptions only for useful recovery or a clearer error.
- Use descriptive names and straightforward steps. Concise code should still be easy to read.
- Add or update focused unit tests for changed behavior and run affected tests. Fix underlying bugs; never weaken, skip, or delete tests just to make them pass.
- Before finishing, remove any code you added that is unused or adds complexity without helping the current task.
