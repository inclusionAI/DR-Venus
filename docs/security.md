# Credential safety before publication

The upstream history contains a tracked `Inference/.env` with non-placeholder
credential values. The personal reproduction layer deletes that file from the
current tree, ignores every populated `.env`, and keeps only `.env.example`
templates. Deleting the current file does **not** remove its historical blobs.

Before this repository is made public:

1. Immediately revoke and rotate every Serper, Jina, summarizer, and Judge key
   that ever appeared in `Inference/.env`. Treat them as compromised even if
   the repository has so far been private.
2. Run the current-tree scan:

   ```bash
   python3 scripts/check_no_secrets.py
   ```

3. Run the history scan. It is expected to fail until history is cleaned:

   ```bash
   python3 scripts/check_no_secrets.py --history
   ```

4. With the repository owner's explicit approval, use `git filter-repo` or BFG
   to remove `Inference/.env` from every ref, then coordinate the required
   force-push and fresh clones with all collaborators. History rewriting is
   destructive and is intentionally not automated here.
5. Run the history scan again and require a zero exit status before publishing.

The scanner reports only commit/path/key categories; it never prints detected
values. Tool and Judge caches also redact configured secrets, but caches and raw
trajectories remain experiment artifacts and should be reviewed before release.
