#!/usr/bin/env python3
"""Write scripts.json: the catalogue's own scripts of the models tier T3 runs (README.md).

  make_scripts_catalogue.py <Models registry checkout> [--out scripts.json] [--all]

Reads the registry's files only (model.yaml, scripts/performance.py, scripts/accuracy.py and the
requirements beside them); imports nothing of the registry and runs no script. Per model with a
package whose row in expected.json is `equal` (--all: every model of catalogue.json) it records what
qualify.py needs to run each script as the registry's own run_model_performance.sh and
run_model_accuracy.sh run it: the accuracy dataset (the first of model.yaml, by its source id
<id>-<variant>, with the pinned sha256 and size), whether a script requires --resources, whether
performance.py requests MLA layer stats, and how accuracy.py constructs its Models (with no options:
the `default` view of expected.json; with options: named in `builds`, judged against the same view
because the tensors go in as the model's own). Hand-written fields of an existing file (`limit`,
`measured_fields`, `notes`) are kept.
"""
import argparse
import ast
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
STDLIB = set(sys.stdlib_module_names)
LOCAL = {'model_report', 'calibration', 'pyneat'}


def read_yaml(path):
    import yaml  # the registry's own format; PyYAML is needed only to regenerate this file
    return yaml.safe_load(path.read_text())


def imports_of(path, seen=None):
    """Modules a script imports that are neither the standard library nor the files beside it."""
    seen = seen if seen is not None else set()
    if path in seen or not path.exists():
        return set()
    seen.add(path)
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split('.')[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split('.')[0]]
        for m in names:
            beside = path.parent / f'{m}.py'
            if beside.exists():
                mods |= imports_of(beside, seen)
            elif m not in STDLIB and m not in LOCAL:
                mods.add(m)
    return mods


def arguments_of(tree):
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument' and node.args:
            flag = node.args[0].value if isinstance(node.args[0], ast.Constant) else None
            required = any(k.arg == 'required' and getattr(k.value, 'value', False) for k in node.keywords)
            if flag:
                out[flag] = required
    return out


def model_constructions(tree):
    """How many pyneat.Model(...) calls pass options, and how many do not."""
    with_options = without = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'Model' \
                and getattr(node.func.value, 'id', '') == 'pyneat':
            if len(node.args) > 1 or node.keywords:
                with_options += 1
            else:
                without += 1
    return with_options, without


BENCHMARK_FIELDS = ('num_samples', 'min_duration_ms', 'include_mla_layer_stats', 'include_mla_stage_stats')


def benchmark_settings(tree):
    """The BenchmarkOptions fields a performance script sets, constants read through module names."""
    constants = {n.targets[0].id: n.value.value for n in tree.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                 and isinstance(n.value, ast.Constant)}
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute) \
                and node.targets[0].attr in BENCHMARK_FIELDS:
            v = node.value
            if isinstance(v, ast.Constant):
                out[node.targets[0].attr] = v.value
            elif isinstance(v, ast.Name) and v.id in constants:
                out[node.targets[0].attr] = constants[v.id]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('registry')
    ap.add_argument('--out', default=str(HERE / 'scripts.json'))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--revision', help='the registry revision to record (default: kept from the existing file)')
    args = ap.parse_args()
    reg = pathlib.Path(args.registry)
    catalogue = json.loads((HERE / 'catalogue.json').read_text())
    expected = json.loads((HERE / 'expected.json').read_text())
    carried = {r['package'] for r in expected['rows'] if r['outcome'] == 'equal'}
    models = sorted({p['model'] for p in catalogue['packages'] if args.all or p['name'] in carried})
    old = json.loads(pathlib.Path(args.out).read_text()) if pathlib.Path(args.out).exists() else {}

    datasets, entries = {}, {}
    for model in models:
        d = reg / 'models' / model
        y = read_yaml(d / 'model.yaml')
        pins = (y.get('accuracy') or {}).get('datasets') or []
        pins = list(pins.values()) if isinstance(pins, dict) else pins
        entry = {}
        if pins:
            pin = pins[0]
            source = f"{pin['id']}-{pin['variant']}"
            entry['dataset'] = source
            ds = datasets.setdefault(source, {'id': pin['id'], 'variant': pin['variant'], 'sha256': pin['sha256'],
                                              'size_bytes': pin['size_bytes'], 'models': 0})
            ds['models'] += 1
            if len(pins) > 1:
                entry['more_datasets'] = [f"{p['id']}-{p['variant']}" for p in pins[1:]]
        modules = set()
        for script in ('performance', 'accuracy'):
            f = d / 'scripts' / f'{script}.py'
            if not f.exists():
                continue
            tree = ast.parse(f.read_text())
            flags = arguments_of(tree)
            s = {}
            if flags.get('--resources'):
                s['resources'] = True
            modules |= imports_of(f)
            if script == 'performance':
                b = benchmark_settings(tree)
                if b:
                    s['benchmark'] = b
            else:
                with_options, without = model_constructions(tree)
                s['view'] = 'default'
                if with_options:
                    s['builds'] = f'{with_options} Model(package, options)' + (f', {without} Model(package)' if without else '')
            entry[script] = s
        entry['modules'] = sorted(modules)
        req = d / 'scripts' / 'requirements-accuracy.txt'
        if req.exists():
            entry['requirements'] = [l.strip() for l in req.read_text().splitlines() if l.strip() and not l.startswith('#')]
        entries[model] = entry

    out = {k: v for k, v in old.items() if k not in ('datasets', 'models')}
    out.setdefault('schema', 1)
    if args.revision:
        out['source'] = {'models_registry': args.revision}
    out['datasets'] = datasets
    out['models'] = entries
    tmp = args.out + '.part'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    pathlib.Path(tmp).replace(args.out)
    print(f'{len(entries)} models, {len(datasets)} datasets -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
