# AAGNet — Module Contract

## Role

This is a **third-party vendor repository** — `AGNet` (Attributed Adjacency Graph
Network) for machining feature recognition on B-Rep solids.

## Boundary

- **IN**: STEP file path (via `AAGExtractor`)
- **OUT**: per-face semantic segmentation, instance segmentation, bottom-face detection
  (raw model outputs via `AAGNetSegmentor` forward pass)

## What lives here

- `dataset/AAGExtractor.py` — STEP → AAG (face/edge attributes + UV grids)
- `models/` — AAGNet model architecture (PNAConv, instance segmentors)
- `utils/data_utils.py` — graph construction, normalization, standardization
- `weights/` — pretrained weights (`weight_on_MFInstseg.pth`) and attribute statistics
- `feature_lists/` — attribute schema (face/edge feature definitions)

## What does NOT happen here

- ❌ Direct modifications to vendor source code
- ❌ Feature instance ownership decisions
- ❌ Gather rules
- ❌ Graph builders or relation rules

## Adapter

New pipeline code calls AAGNet **through an adapter only**:

```
src/occline/aagnet/per_face.py
```

The adapter:
1. Imports AAGNet modules by adding this directory to `sys.path`
2. Runs inference via `AAGNetSegmentor`
3. Returns **per-face semantic evidence** (NOT native AAGNet feature groups)

## Architecture Note

The old pipeline used AAGNet's native `feature_id` assignment (connected-component
groups from instance segmentation) as the **final feature instance ownership**.

In the new architecture, AAGNet provides only **per-face semantic evidence**
(e.g., "face_3 is likely part of a blind_hole"). The decision of which specific
faces belong to which feature instance is made by the **v5.1 leaf-class
recognisers** in `src/features/`.

This means: AAGNet native feature groups → discarded at the adapter boundary.
Only per-face logits / predicted labels are forwarded.
