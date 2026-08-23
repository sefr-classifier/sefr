# Publishing to GitHub

These steps update [sefr-classifier/sefr](https://github.com/sefr-classifier/sefr) from this folder.

## One-time setup

```bash
git clone https://github.com/sefr-classifier/sefr.git ../sefr-upstream
cd ../sefr-upstream
```

## Sync and push

```bash
rsync -av --delete \
  --exclude '.git' \
  --exclude '.DS_Store' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.egg-info/' \
  ./ ../sefr-upstream/

cd ../sefr-upstream
git add -A
git status
git commit -m "Refresh repo: package layout, docs, CI"
git push origin master
```

## What changed vs the old repo

| Before | After |
|--------|-------|
| Single `SEFR.py` + 3-line README | Installable `sefr` package + `SEFR.py` shim |
| No tests / CI | pytest + GitHub Actions |
| — | `examples/`, `CITATION.cff`, `pyproject.toml` |

The original [`arduino-c`](arduino-c) file is preserved.
