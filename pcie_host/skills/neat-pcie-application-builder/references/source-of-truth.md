# Source Of Truth

Use the API installed on the customer host. The PCIe host API is versioned and packaged separately
from the regular Neat Library, so SDK headers and examples can describe a different surface.

## Preferred Order

1. Installed C++ header:
   `/usr/include/simaai/neat/pcie/Model.h`
2. Installed Python module in the selected Python environment:
   `pyneatpcie`
3. Packaged PCIe tutorials, when the PCIe extras bundle is available:
   `share/sima-pcie-host/tutorials/`
4. In a Neat core source checkout only:
   - `pcie_host/include/simaai/neat/pcie/Model.h`
   - `docs/develop-apps/development-workflow/pcie-model.mdx`
   - `pcie_host/tutorials/`

Do not use the regular core `include/model/Model.h`, `pyneat.Model`, or DevKit tutorials to infer
PCIe host behavior.

## Public Surface

The intended application surface is the content of `Model.h`:

- `ConnectionOptions`
- `ModelOptions` and its preprocessing/decode enums
- `Tensor`, `TensorList`, `TensorInfo`, and `ModelInfo`
- `TensorDType`, `TensorLayout`, image formats, planes, and routes
- `Model`

`Runtime.h` is deliberately outside this skill. Do not use symbols that happen to be exported by
the Python extension if they belong to `Runtime.h`.

## Inspect The Installed Release

For C++:

```bash
test -r /usr/include/simaai/neat/pcie/Model.h
rg -n "^(struct|class|enum class|using) " /usr/include/simaai/neat/pcie/Model.h
```

For Python, use the interpreter that owns the installed wheel. The PCIe package installer uses
`~/pyneatpcie/bin/python` by default when Python support is requested:

```bash
~/pyneatpcie/bin/python -c \
  'import pyneatpcie as p; print(p.__version__); print(p.Model); print(p.Tensor)'
```

Nanobind callables do not always expose useful `inspect.signature()` output. Prefer `help()`,
`dir()`, the installed header, and a minimal import/run check.

## Model Contract

Use `Model::info()` or `Model.info()` to obtain the inference tensor names, dtypes, shapes, and byte
sizes recorded in the model archive. `ModelInfo` does not incorporate runtime preprocessing or
postprocessing options:

- In tensor mode, `info().inputs` is the contract for constructing model-ready input tensors.
- In image mode, `info().inputs` describes the model ingress that card-side preprocessing must
  produce. It does not describe the decoded image payload submitted by the host; that payload's
  format and geometry come from the image and preprocessing configuration.
- `info().outputs` always describes the archive's inference outputs. For example, enabled boxdecode
  replaces those raw outputs at runtime with one `UInt8` tensor whose route name is `BBOX`.

Do not derive an application contract from plugin-private JSON files inside the archive. If
contributor-level archive inspection is unavoidable, use only `mpk.json` or `*_mpk.json` as the
model contract.
