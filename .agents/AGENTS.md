# Python Script Execution

Always use `uv run` to execute Python scripts in this workspace (e.g., `uv run <script>.py`). Do not execute them directly using `python <script>.py` or `.venv/Scripts/python.exe <script>.py`.

# Release Process & Automation

When asked to prepare, draft, or execute a release of `core-lib` (e.g., "Prepare the next release", "Draft release vX.Y.Z", "Update release notes and version", "Release this lib in GitHub"):

Follow a strict two-phase workflow with a mandatory user review gate.

### Phase 1: Draft & Prepare (Never publish automatically)

1. **Run Tests**:
   - Verify all tests pass before preparing the release:
     ```powershell
     uv run pytest
     ```

2. **Analyze Git History**:
   - Check commits since the latest git tag:
     ```powershell
     git describe --tags --abbrev=0
     git log $(git describe --tags --abbrev=0)..HEAD --oneline
     ```
   - Categorize changes: breaking/behavioral changes, new features, bug fixes, dependency updates, and documentation.

3. **Version Bump**:
   - Determine the target Semantic Version (`vX.Y.Z`).
   - Update both package definition files:
     - `pyproject.toml`: update `version = "X.Y.Z"` under `[project]`
     - `setup.py`: update `version='X.Y.Z'` in `setup(...)`

4. **Update Master Release Notes**:
   - Append the new release section in `RELEASE_NOTES.md` under the current month and year.
   - Follow the established format:
     ```markdown
     ### vX.Y.Z - <Brief Title / Main Focus> (Month DD, YYYY)

     #### <Key Category / Behavioral Changes>
     - Detailed bullets outlining changes, API updates, or behavioral notes.
     - Note downstream impact (e.g., for `mcp-doc-qa`, `agent-rfx`, `saas-admin`) if applicable.
     ```

5. **Update Task Tracking**:
   - Review `TODO.md` and check off or remove items that have been completed.

6. **MANDATORY REVIEW GATE**:
   - Stop execution and present the draft version bump, summary of changes, and release notes to the user for review.
   - Do **NOT** commit, push, create tags, or publish GitHub releases until the user gives explicit approval.

### Phase 2: Publish & Tag (Only after explicit user approval)

1. **Commit & Push Changes**:
   ```powershell
   git add pyproject.toml setup.py RELEASE_NOTES.md TODO.md
   git commit -m "chore(release): vX.Y.Z"
   git push origin main
   ```

2. **Create and Push Git Tag**:
   ```powershell
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. **Create GitHub Release**:
   - Create the release using the GitHub CLI (`gh`):
     ```powershell
     gh release create vX.Y.Z --repo Aiuj/core-lib --title "vX.Y.Z - <Title>" --notes-file <path-to-notes>
     ```
   - Alternatively, pass the markdown notes inline via `--notes "<markdown_body>"`.
   - Verify the release is visible via `gh release view vX.Y.Z --repo Aiuj/core-lib` or `gh release list -R Aiuj/core-lib -L 3`.

4. **Downstream Consumers Notification**:
   - Remind the user if consuming services (`agent-rfx`, `mcp-doc-qa`, `saas-admin`, etc.) need to update their dependency reference or lockfile to point to the new release tag.
