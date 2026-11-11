# Python Script Execution

Always use `uv run` to execute Python scripts in this workspace (e.g., `uv run <script>.py`). Do not execute them directly using `python <script>.py` or `.venv/Scripts/python.exe <script>.py`.

# Context Hygiene

- Never read `uv.lock` directly (~460KB); use `uv tree` or `uv pip show <pkg>` instead.
- Never open `.coverage` (binary SQLite file).
- Grep `README.md` and `RELEASE_NOTES.md` for specific sections instead of reading whole files. When drafting a release, read only the current month's section.
- Grep topic-specific guides in `docs/` rather than scanning the directory wholesale.

# Release Process & Automation

When asked to prepare, draft, or execute a release of `core-lib`:

Follow a strict two-phase workflow with a mandatory user review gate.

### Phase 1: Draft & Prepare (Never publish automatically)
1. **Run Tests**: Verify all tests pass: `uv run pytest`.
2. **Analyze Git History**: Check commits since latest tag:
   `git describe --tags --abbrev=0`
   `git log $(git describe --tags --abbrev=0)..HEAD --oneline`
3. **Version Bump**: Update `version = "X.Y.Z"` in both `pyproject.toml` and `setup.py`.
4. **Update Master Release Notes**: Append section to `RELEASE_NOTES.md` under current month. Note downstream impact (`mcp-doc-qa`, `agent-rfx`, `saas-admin`) if applicable.
5. **Update Task Tracking**: Review `TODO.md` and check off completed items.
6. **MANDATORY REVIEW GATE**: Present draft bump and release notes to user for review. Do NOT commit, tag, or publish without explicit approval.

### Phase 2: Publish & Tag (Only after explicit user approval)
1. **Commit & Push**:
   `git add pyproject.toml setup.py RELEASE_NOTES.md TODO.md`
   Review the entire staged index (git diff --cached), including any pre-existing staged changes. Stage all intended release changes before calculating the digest.
   `uv run --no-sync scripts/release_integrity.py create`
   `git add RELEASE_INTEGRITY.json`
   `git commit -m "chore(release): vX.Y.Z"`
   `uv run --no-sync scripts/release_integrity.py verify HEAD`
   If verification fails, stop and regenerate the manifest from the corrected staged release before committing again.
   `git push origin main`
2. **Create and Push Git Tag**:
   `git tag vX.Y.Z`
   `uv run --no-sync scripts/release_integrity.py verify vX.Y.Z`
   `git push origin vX.Y.Z`
3. **Create GitHub Release**:
   `gh release create vX.Y.Z --repo Aiuj/core-lib --title "vX.Y.Z - <Title>" --notes-file <path-to-notes>`
4. **Notify Downstream**: Remind user if consuming services need lockfile updates.

### Release integrity

`RELEASE_INTEGRITY.json` is committed with each release. It records a SHA-256 digest over all tracked paths, Git file modes, and blob contents in the staged release tree, excluding only the manifest itself. Run `uv run --no-sync scripts/release_integrity.py verify vX.Y.Z` to check a tag later; the command uses Git objects, not the current working tree. Any staged change after creating the manifest requires regenerating and staging it again. The digest covers source tracked by Git, not untracked files, generated release assets, or submodule contents (submodules are rejected). A digest stored beside the content detects accidental changes but does not authenticate a maliciously replaced commit or tag; establish tag authenticity separately, for example with a trusted signed tag.
