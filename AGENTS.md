# Agent notes

The notes for AI agents in this repository are in **[CLAUDE.md](CLAUDE.md)**,
whatever agent you are.

Deploying for the first time? **[install/SETUP.md](install/SETUP.md)** is the
procedure, and it
ends with the viewer running and the browser open on it.

Two things worth knowing before you start, in case you read no further:

- If an install step fails, send the output rather than working around it. The
  package versions in `install/environment.yml` were verified together, and a
  substituted one — `dgl` above all — can install cleanly and answer wrongly.
- `nx: NOT_BUILT` is not a failure. It means this machine has no licensed
  Siemens NX, which is expected and does not affect the open-source results.
