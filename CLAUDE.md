## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Do not create audit, TODO, plan, or summary files unless the user explicitly asks for a file.
- Don't "improve" adjacent code, comments, or formatting.
- Keep code, comments, test names, test fixture text, and non-user-facing strings in English.
- User-facing copy may use the product's target language, but internal diagnostics and tests stay English.
- Use fake or randomized test emails, IPs, URLs, and tokens; never real personal or service values.
- Do not add native hover tooltip text such as `title` attributes by default; only add them when explicitly requested.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

When debugging with logs:
- Do not add enable/disable switches unless explicitly requested.
- When the user says the issue is fixed, do a broader check for similar failure paths before closing.
- If a fix took several attempts, review the full diff and revert any leftover workaround that no longer belongs.
- After the fix is verified, remove temporary diagnostic logs and test noise.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Keep Files Concise

**Target a maximum of 300 lines per file.**
Test files may be longer when the extra length is meaningful coverage, but keep them under 500 lines.
If a non-test file exceeds 300 lines, or a test file exceeds 500 lines, propose a logical split into smaller modules.

## 6. Commit Messages

**Use emoji + conventional keyword + scope.**

Commit only when explicitly requested. Wrap-up or verification requests are not commit permission.

When creating commits:
- Only commit when the user explicitly asks for a commit.
- Format commit messages as `emoji type(scope): subject`.
- Keep the subject short, imperative, and specific.
- Match the type to the change, such as `fix`, `feat`, `refactor`, `test`, `docs`, `chore`.
- Example: `🐛fix(shortcuts): block background shortcuts while dialogs are focused`

## 7. Merge Conflicts

**Prefer the latest side, but still reason through the merge.**

When resolving conflicts:
- Treat incoming/latest as the default baseline, not as a blind overwrite rule.
- Compare both sides for behavior, imports, types, tests, and surrounding dependencies.
- Preserve the latest intent unless it breaks the current main branch or drops a necessary existing fix.
- If the latest side is stale against main, apply the smallest compatibility fix instead of keeping both versions.
- After resolving, check for conflict markers, unmerged paths, compile or type issues, and run the most relevant tests when practical.

## 8. Git Branch Workflow

**main is always runnable. All work happens on feature branches.**

```
main ────────────────────────────────────── (protected, always green)
  \
feat/<phase>-<desc> ── ● ── ● ── merge + tag
  \
fix/<desc> ─────────── ● ── merge + tag
```

Rules:
- **Never push directly to main.** All changes go through a feature branch.
- Branch naming: `feat/<phase>-<short-desc>` for features, `fix/<short-desc>` for fixes.
- Squash-merge into main when the branch is verified runnable.
- Delete the feature branch after merge.
- Run the relevant scripts (inference, tests) before pushing any branch.

## 9. Semantic Versioning

**Tag every merge to main with MAJOR.MINOR.PATCH.**

```bash
git tag -a vX.Y.Z -m "<emoji> vX.Y.Z: <one-line summary>"
git push origin vX.Y.Z
```

Rules:
- **MAJOR (X):** First deployment-ready release (1.0.0). Rarely bumped thereafter.
- **MINOR (Y):** Each completed Phase from docs/task_todo.md.
- **PATCH (Z):** Bug fixes, refactors, dependency updates that don't add features.

Current version map:
| Version | Phase | Milestone |
|---------|-------|-----------|
| v0.1.0 | 0-3 | Inference pipeline working |
| v0.2.0 | 4 | Custom dataset ready |
| v0.3.0 | 5 | GPU training complete |
| v0.4.0 | 6 | ONNX export working |
| v1.0.0 | 7 | FastAPI deployment ready |

## 10. Commit Emojis

| Type | Emoji | When |
|------|-------|------|
| feat | ✨ | New feature or script |
| fix | 🐛 | Bug fix |
| docs | 📝 | Documentation only |
| refactor | ♻️ | Restructure without behavior change |
| test | ✅ | Add or update tests |
| chore | 🔧 | Config, deps, tooling |
| init | 🎉 | First commit of a new project/module
