# pyneat Patterns

Use this reference when the user asks for Python application code with Neat.

## Primary reference locations

- Core Python package source:
  - `/neat-resources/core-src/python/pyneat` in eLxr SDK setups
  - `core/python/pyneat` in repo checkouts
- Core Python tutorials:
  - `core/tutorials/*/*.py`
- Apps Python examples:
  - `/neat-resources/apps-src/examples/**/python/main.py` in eLxr SDK setups
  - `apps/examples/**/python/main.py` in repo checkouts

## Preferred generation patterns

- Start with public `pyneat` only.
- For minimal validation, import `pyneat` and construct `Graph`, `Model`, or `Tensor` directly.
- For image/model flows, use `pyneat.ModelOptions()`, `pyneat.Tensor.from_numpy(...)`, and explicit `PixelFormat`.
- For run loops, build `Graph`, set `RunOptions`, and choose output/overflow policy explicitly.

## Runtime assumptions

- On DevKit, activate `$HOME/pyneat/bin/activate`.
- In SDK workflows, `dk /workspace/path/to/script.py` can run Python scripts on the connected DevKit and will try to activate `pyneat`.

## Avoid

- Importing private implementation modules directly unless the task is explicitly about binding internals.
- Assuming `pyneat` and C++ APIs are perfectly one-to-one; confirm against Python tutorials/examples first.
