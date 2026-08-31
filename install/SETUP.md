# Deployment

Follow this top to bottom. Each step has a check. At the end the pipeline has
processed the sample part, the viewer is running, and the browser is open on
it — ready to use.

Prerequisites: Windows 10 or 11, conda (`conda --version` must answer), about
3 GB of disk. No GPU needed.


## 1 · Create the environment

```
conda env create -f install/environment.yml
```

Several minutes, roughly 300 MB downloaded.

Then find the environment's interpreter once and use it by full path from here
on — that removes the commonest deployment mistake, which is running from a
different environment without noticing:

```
conda run -n CNCgAIde python -c "import sys; print(sys.executable)"
```

Below, `PY` stands for that path, e.g.
`C:\Users\you\miniconda3\envs\CNCgAIde\python.exe`.

**Run every command in this file from the repository root**, not from this
`install/` folder - the paths below are written relative to the root.


## 2 · Check the environment

```
set PYTHONNOUSERSITE=1 && "PY" -c "import torch, dgl, numpy, scipy, OCC, yaml; from OCC.Core.STEPControl import STEPControl_Reader; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('numpy', numpy.__version__); print('OCC ok')"
```

Expect `torch 2.0.1+cpu | cuda False`, `numpy 1.26.4`, `OCC ok`. `cuda False`
is correct — CPU is the supported configuration and gives the same results.

Keep `PYTHONNOUSERSITE=1` set for the rest of this file; it stops an old
`pip install --user` package from shadowing the environment.

```
"PY" -m compileall -q src run_pipeline.py
```

Silence means success.


## 3 · Process the sample part

```
set PYTHONPATH=src && "PY" run_pipeline.py refresh --input inputs --output-tag latest --device cpu
```

```
set PYTHONPATH=src && "PY" install/verify_install.py
```

This must print **`INSTALL VERIFIED`**. It compares the run against the result
the sample part is known to produce — how many features of each kind were
found, how many relations, how many rule verdicts.

That check exists because "it ran without an error" proves very little here. An
environment that is subtly wrong still recognises flat faces and simple holes
correctly and quietly gets chamfers and fillets wrong. The sample part carries
a chamfer and two fillets for exactly that reason.

If it prints anything else, stop and send the output — see
[When something fails](#when-something-fails).


## 4 · Start it

```
set PYTHONPATH=src && "PY" run_pipeline.py view --output-tag latest
```

The viewer starts on http://localhost:8765 and opens a browser. You should see
the parts list with the sample part in it; clicking it opens the 3D model with
its features coloured.

**Leave it running.** This is the finished state: from here everything happens
in the browser — adding parts, re-running them, comparing them. `README.md`
explains the interface.

To stop it later:

```
set PYTHONPATH=src && "PY" run_pipeline.py view --stop --port 8765
```


## Optional: Siemens NX

Only if this machine has NX installed and licensed. Without it the open-source
recognition line runs exactly as before — a missing NX is recorded as
"not built" with the reason, not as an error.

Find the NX installation root (the folder containing `NXBIN\run_journal.exe`
and `UGII\customization_base.xml`), copy `config\nx_local.example.json` to
`config\nx_local.json`, and put the path in it:

```json
{ "nxRoot": "C:/Program Files/Siemens/NX2506" }
```

Forward slashes are fine and avoid escaping problems. Re-run refresh; the NX
line builds alongside the open-source one, and the viewer gets a second line to
switch to. Developed against NX2506; the feature schema is read from your
install, so what NX declares is what you get.


## Optional: GPU

Not needed. Recognition results on CPU are identical to CUDA — same features,
same relations, same verdicts — and the heaviest reference part takes about 16
seconds.

If you want the speed, swap torch and dgl for their CUDA builds *after* the CPU
install verifies, as a separate step. A torch/dgl CUDA mismatch installs with no
error at all and then fails on the first inference, which is a far worse problem
to diagnose than a slow run.


## When something fails

Send the command, the complete output, and the step number. During the install
that is more useful than a workaround: the package versions in
`install/environment.yml` were verified together, and a substituted one can
install
cleanly and answer wrongly — `dgl` above all, whose current release is 2.x and
is incompatible with `torch 2.0.1` and with the vendored network code.

**`conda env create` cannot solve the environment.**
Same packages, explicitly, in this order. The order matters: the CAD kernel is
on conda-forge only, and letting the solver mix channels is the classic cause of
an OCCT that imports and then crashes.

```
conda create -n CNCgAIde python=3.10 -c conda-forge --override-channels -y
```

```
conda install -n CNCgAIde "pythonocc-core=7.9.3=novtk*" -c conda-forge --override-channels -y
```

```
set PYTHONNOUSERSITE=1 && "PY" -m pip install torch==2.0.1 dgl==1.1.0 numpy==1.26.4 scipy==1.13.1 timm==1.0.27 networkx==3.4.2 tqdm==4.68.3 PyYAML==6.0.3
```

**`import OCC` fails or crashes the interpreter.** Almost always a mixed-channel
install. `conda env remove -n CNCgAIde` and redo it with `--override-channels`
rather than repairing in place.

**`import dgl` fails, or inference crashes.** Check the version is exactly
1.1.0 (`"PY" -m pip show dgl`).

**`import numpy` fails inside torch.** numpy 2.x; reinstall `numpy==1.26.4`.

**Packages resolve but behave oddly.** Check for a leaking user-level
site-packages:
`"PY" -c "import site; print(site.ENABLE_USER_SITE, site.getusersitepackages())"`.

**`install/verify_install.py` reports a mismatch.** Send the output as it is.
Which
numbers moved says which part of the environment is wrong.

**The port is in use.** The viewer will not move silently. `view --list` shows
what is running, `view --stop --port N` stops it, `--port 8766` uses another.
