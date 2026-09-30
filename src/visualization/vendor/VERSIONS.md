# Vendored browser libraries

Which build of each library is actually in this directory. `NOTICE` at the
repository root records the licences; this file records the *identity*, so a
security advisory or an upstream fix can be matched against what is here, and
so a re-download can be checked rather than trusted.

Nothing in this directory is patched: `grep -rni "gatherseed\|cncgaide\|patched"`
over it returns nothing, and the sha256 below is the whole file as downloaded.

The version column says where the number comes from. **Declared in the file**
means the file itself states it and the number can be re-read at any time.
**Not declared** means neither the file nor the upstream bundle carries a
version string: for those the sha256 *is* the identity, and the way to learn
the version is to hash the published releases until one matches. They are not
guessed here.

| File | Version | Where the version comes from | Upstream |
| --- | --- | --- | --- |
| `three.module.js` | r160 | declared in the file (`REVISION = '160'`) | https://github.com/mrdoob/three.js |
| `OrbitControls.js` | r160 | same distribution, `examples/jsm/controls/OrbitControls.js` | https://github.com/mrdoob/three.js |
| `cytoscape.min.js` | 3.28.1 | declared in the file (`version="3.28.1"`), and `cytoscape.version` in a browser agrees | https://github.com/cytoscape/cytoscape.js |
| `cytoscape-fcose.js` | not declared | webpack UMD bundle with no version string | https://github.com/iVis-at-Bilkent/cytoscape.js-fcose |
| `cose-base.js` | not declared | webpack UMD bundle with no version string | https://github.com/iVis-at-Bilkent/cose-base |
| `layout-base.js` | not declared | webpack UMD bundle with no version string | https://github.com/iVis-at-Bilkent/layout-base |

The three fCoSE files are one dependency chain: `cytoscape-fcose` requires
`cose-base`, which requires `layout-base`. Replacing one means replacing all
three.

## sha256

Recompute with:

```
python -c "import hashlib,pathlib;[print(f'{p.name}  {hashlib.sha256(p.read_bytes()).hexdigest()}') for p in sorted(pathlib.Path('src/visualization/vendor').glob('*.js'))]"
```

| File | Bytes | sha256 |
| --- | ---: | --- |
| `cose-base.js` | 118906 | `7cae9509bd36235a63a85e71c8d9fa2cd0bc1d0c1ecc5b5a737976f39d040ddf` |
| `cytoscape-fcose.js` | 57239 | `4b1cab218d74996aa59cd8473f9239cc6398b8c1774d84d7e59ad9a68959cb57` |
| `cytoscape.min.js` | 365827 | `92d752b48ea949720675865197fd2a0001c95bc5888545e990af60321712d4c6` |
| `layout-base.js` | 147958 | `ec15ab5df9af3f20708f4faab994accf91cda71848cd5bb10a23432cc50b6745` |
| `OrbitControls.js` | 29868 | `5a44a9e86a2a0fb11933eed69bc2cd33c76a496854c1aed6ed776efa87d7b064` |
| `three.module.js` | 1272972 | `76dea8151bc9352aef3528b4262e249b2604f62543828328db978d060d61a495` |
