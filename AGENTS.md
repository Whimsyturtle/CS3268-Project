# Agent instructions

## Project context

- Start with [README.md](README.md), then read the source documents relevant to the task.
- Follow the experiment design in the [proposal](CS3268_Project_Proposal_Group_6.md) exactly. Any deviation requires explicit user approval.
- Follow the [course requirements](docs/CS3268_Project_Document.md) and use the [cluster guide](docs/SoC_Cluster_Guide.md) for cluster work.
- All new cluster construction, tuning, and experiment runs must follow the [shared runtime requirements](docs/Cluster_Runtime.md), including new steering methods.

## Workflow and verification

- Follow the [contribution workflow](README.md#contributing) when submitting changes.
- Before major work, read the relevant issue, its dependencies, and linked PRs. Check the actual files and verification evidence before treating work as complete; an issue does not authorize changes to the proposal.
- Keep related issues current under the contribution workflow, including blockers, verification, and any required reruns. If GitHub is unavailable, report the pending update rather than claiming it was made.
- **Plan coherent changes.** The parent agent evaluates the request and divides the work into small changes, each with one purpose and possibly spanning multiple files. Identify dependencies before starting.
  - Keep edits within the request. Inspect actual files before assuming scripts, commands, or results exist.
- **Delegate when useful.** Use subagents when they clearly save time or improve results. Give them clearly defined tasks that can be done independently, share relevant repository instructions, and avoid overlapping edits. Prefer GPT-6.1 Sol or Opus 5.5 when available.
- **Implement and verify.** The parent agent reviews and combines delegated work, then runs appropriate checks for each change.
- **Obtain independent review.** Before completing each change, obtain a review from a new **independent reviewer subagent** that neither implemented it nor inherited the implementation conversation. Review before dependent work if errors could invalidate it.
  - Provide the request, explicit approvals, diff, relevant repository instructions and source documents, and verification evidence.
  - Reviewers must check **internal consistency**: the change aligns with the request, works as intended, and appropriate verification supports it. They must also check **external consistency**: the change aligns with repository rules and project requirements, including this file, README.md, and the proposal.
  - Reviewers must remain read-only. Report concrete defects, missing necessary checks, or rule violations, with an affected location and explanation. Recommend the smallest necessary correction. Reporting no findings is valid.
  - For large changes, divide review into distinct areas and check their interactions before completion.
- **Resolve findings and finish.** The parent agent must assess findings, fix confirmed issues, run affected checks, and obtain targeted review of substantive fixes before drawing conclusions. The same reviewer may recheck fixes within that change. Check the combined result, report checks and limitations, and mark unchecked results.
  - If bugs invalidate results, fix the bugs and rerun affected experiments when possible; otherwise provide rerun instructions. Keep affected conclusions pending until replacement results have been checked.

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
