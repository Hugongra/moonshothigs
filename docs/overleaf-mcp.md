# Overleaf integration

We use Overleaf in **two** independent ways:

| Purpose                                  | Mechanism                                                   | File                            |
| ---------------------------------------- | ----------------------------------------------------------- | ------------------------------- |
| **Push** the rendered bundle every run   | Overleaf Git integration via Python subprocess              | `scripts/sync_overleaf.py`      |
| Read / inspect the cloud project in Cursor (optional) | [OverleafMCP](https://github.com/mjyoo2/overleafmcp) (Node) | `~/.cursor/mcp.json` entry      |

The two layers solve different problems. The MCP server only exposes **read** tools
(`list_files`, `read_file`, `get_sections`, `status_summary`), so it cannot
update the Overleaf project. For automated sync we drive Overleaf's underlying
Git repository directly.

---

## 1. Auto-push every pipeline run

### One-time setup

```bash
cp configs/overleaf.example.yaml configs/overleaf.local.yaml
# edit the new file and paste:
#   project_id: <UUID from https://www.overleaf.com/project/<id>>
#   git_token:  <Overleaf → Account Settings → Git Integration → Create Token>
```

`configs/overleaf.local.yaml` is gitignored. You can alternatively set
`OVERLEAF_PROJECT_ID` and `OVERLEAF_GIT_TOKEN` env vars (they take precedence
over the YAML).

### Trigger sync with the pipeline

```bash
# Flag form
python run_neurips_pipeline.py --reuse-eval-json evals/results/neurips_full.json \
    --no-compile --sync-overleaf

# Env form (useful for CI / the shell wrapper)
OVERLEAF_SYNC=1 scripts/build_rag_and_paper.sh
```

Dry-run without committing:

```bash
python run_neurips_pipeline.py --sync-overleaf --overleaf-dry-run
```

### What it does

1. Clones `https://git:<token>@git.overleaf.com/<project_id>` into
   `./.overleaf_mirror/` (or `git fetch` + hard-reset if it already exists).
2. Copies **everything** under `output/neurips_overleaf_bundle/`
   (`main.tex` + `figures/`) into the mirror, overwriting.
3. If the working tree is clean, exits silently (`skip_if_unchanged: true`).
4. Otherwise `git add -A` → `git commit -m "auto: regenerate from pipeline (<sha> @ <ts>)"`
   → `git push origin HEAD:master`.

The sha in the commit message is the first 10 chars of `sha256(main.tex)` so
you can tell runs apart, and Overleaf's real-time view updates as soon as the
push completes.

If sync fails (network, auth, merge conflict on the remote), the pipeline
logs the error but **does not abort** — the local bundle under
`output/neurips_overleaf_bundle/` is still valid and can be uploaded manually.

### Manual one-off push

```bash
python scripts/sync_overleaf.py
python scripts/sync_overleaf.py --dry-run
python scripts/sync_overleaf.py --config configs/overleaf.alt.yaml
```

---

## 2. OverleafMCP (read-only inspection, optional)

Handy if you want Cursor's Composer / Chat to answer "what's currently in the
Introduction section of the Overleaf project?" without leaving the editor.

```bash
git clone https://github.com/mjyoo2/overleafmcp.git ~/tools/OverleafMCP
cd ~/tools/OverleafMCP && npm install
cp projects.example.json projects.json
# edit projects.json with { default: { name, projectId, gitToken } }
```

Then add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "overleaf": {
      "command": "node",
      "args": ["/Users/uzzielperez/tools/OverleafMCP/overleaf-mcp-server.js"]
    }
  }
}
```

Restart Cursor. The MCP exposes `list_files`, `read_file`, `get_sections`,
`get_section_content`, and `status_summary` against the configured Overleaf
project — strictly read-only. Writes still go through
`scripts/sync_overleaf.py`.

---

## Security

- `configs/overleaf.local.yaml`, `.overleaf_mirror/`, and the MCP's
  `projects.json` are all gitignored.
- Tokens are embedded in the remote URL for `git` and never logged (only a
  masked form `https://git:***@git.overleaf.com/<project>` appears in output).
- Rotate the Overleaf Git token whenever you share a machine or repo.
