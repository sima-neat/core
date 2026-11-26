#!/usr/bin/env python3
"""qualify.py: the qualification harness of the Dataflow backend (README.md).

Per catalogue package and option view the outcome on the Dataflow backend is exactly one of two:
  equal        every output byte-equal to GStreamer's through the customer's calls, and every
               customer-visible field equal, except the named rows of expected_differences.json,
               which must hold exactly as written;
  not_carried  the first build fails with the one error form
               "[<code>] <where>: <subject> is not carried by the Dataflow backend: <fact>.\nHint: ..."
               whose hint names simaai::neat::set_backend(...GStreamer), with the subject and fact
               expected.json records, and the same program runs on GStreamer.
Anything else fails. expected.json says which outcome each row has; a change that widens what
Dataflow carries flips its rows in the same commit.

Subcommands
  census    T0: run neat_qualification_census over a corpus, check every verdict against
            expected.json (and the error form of every refusal), optionally against another census.
  compare   T1/T2: per row, the comparator on GStreamer, again on GStreamer (the determinism floor)
            and on Dataflow, judged against expected.json and expected_differences.json.
  judge     the same judgement from records kept by an earlier `compare --keep`.
  seed      write expected.json from a census (rows keep their hand-written `forms` overrides).
  rows      list the rows a selection names (tiers: catalogue.json "tiers").
  scripts   T3: the registry's own performance.py and accuracy.py, unchanged, per package on each
            backend through launch_script.py (scripts.json), judged against expected.json: the
            reports' fields and the accuracy metrics equal, or the one refusal. --judge-only
            judges runs kept by earlier invocations.
  speed     the latency and throughput table of the performance.py reports a `scripts` run kept.
  selftest  the judge's own rules on synthetic records (no device, no package).
"""
import argparse
import collections
import gzip
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
CATALOGUE = HERE / 'catalogue.json'
SCRIPTS = HERE / 'scripts.json'
LAUNCHER = HERE / 'launch_script.py'
EXPECTED = HERE / 'expected.json'
DIFFERENCES = HERE / 'expected_differences.json'

# The one form (src/pipeline/internal/DataflowNotCarried.h), read from the outside.
ONE_FORM = re.compile(
    r'^\[(?P<code>[a-z_]+\.[a-z_]+)\] (?P<where>[A-Za-z_]+::[A-Za-z_]+(?:\([^)]*\))?): '
    r'(?P<subject>.+?) is not carried by the Dataflow backend(?P<board> on this board)?: '
    r'(?P<fact>.+)\.\nHint: (?P<hint>.+)$', re.S)
HINT_CALL = 'simaai::neat::set_backend(simaai::neat::RunBackend::GStreamer)'
CODES = {'misconfig.pipeline_shape', 'misconfig.option_out_of_range', 'misconfig.input_shape',
         'infra.dispatcher_unavailable'}
# Words the standard keeps out of a fact (plan-entry.md 4.4.6): no promise, no plan step.
FORBIDDEN_IN_FACT = re.compile(r'\byet\b|\bnot supported\b|\bRunBackend::Auto\b')
FORMS = ('graph', 'model', 'model_run', 'plan', 'bench')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    tmp = str(path) + '.part'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write('\n')
    os.replace(tmp, path)


def parse_one_form(text):
    """The parts of a not-carried error, or None when `text` is not in the one form."""
    m = ONE_FORM.match(text or '')
    if not m or HINT_CALL not in m.group('hint'):
        return None
    d = m.groupdict()
    d['board'] = bool(d['board'])
    return d


def check_sentence(code, subject, fact):
    """The standard's rules for a planner's refusal; a list of what breaks them."""
    bad = []
    if code not in CODES:
        bad.append(f'code {code!r} is not one of the four families')
    if not subject:
        bad.append('empty subject')
    if not fact:
        bad.append('empty fact')
    if fact and FORBIDDEN_IN_FACT.search(fact):
        bad.append(f'fact has a forbidden word: {FORBIDDEN_IN_FACT.search(fact).group(0)!r}')
    return bad


# ---- the catalogue and its rows --------------------------------------------------------------------
def catalogue():
    return load_json(CATALOGUE)


def component_views(cat):
    """(package, view, component) for every row: the app view uses the component's script options,
    the default view none."""
    for p in cat['packages']:
        for view in ('app', 'default'):
            yield p, view


def options_lines(cat, comp, view):
    if view == 'default' or not comp.get('options'):
        return None
    return cat['options'][comp['options']]


def write_options(cat, comp, view, directory):
    lines = options_lines(cat, comp, view)
    if lines is None:
        return '-'
    path = pathlib.Path(directory) / f"{comp['options']}.opts"
    if not path.exists():
        path.write_text('\n'.join(lines) + '\n')
    return str(path)


def select_rows(expected, args):
    rows = expected['rows']
    out = []
    names = set(args.names.split(',')) if getattr(args, 'names', None) else None
    tier = getattr(args, 'tier', None)
    tier_set = None
    if tier:
        tiers = expected.get('tiers', {})
        if tier not in tiers:
            sys.exit(f'no tier {tier} in expected.json')
        tier_set = {(r['package'], r['view']) for r in tiers[tier]['rows']} if 'rows' in tiers[tier] else None
    for r in rows:
        if names and r['package'] not in names:
            continue
        if getattr(args, 'view', None) and args.view != 'both' and r['view'] != args.view:
            continue
        if getattr(args, 'outcome', None) and args.outcome != 'all' and r['outcome'] != args.outcome:
            continue
        if tier_set is not None and (r['package'], r['view']) not in tier_set:
            continue
        out.append(r)
    return out


# ---- census (T0) ---------------------------------------------------------------------------------------
def read_census(path):
    recs = {}
    for line in open(path):
        line = line.strip()
        if not line.startswith('{'):
            continue
        r = json.loads(line)
        if r.get('record') == 'census':
            recs[(r['id'], r['view'])] = r
    return recs


def expected_refusal(comp, form):
    """The refusal expected.json records for a refused component under one entry form: the
    component's own (the planner's verdict on the route's stages, which the census checks), unless
    an entry of its `form_refusals` names the form. A form that refuses at another point states its
    own there: a Model of several inputs is refused when the Graph of its stages is qualified,
    before the stages are planned, so its model, model_run, plan and bench forms name the model
    where the graph form names the stage."""
    for r in comp.get('form_refusals', []):
        if form in r.get('forms', []):
            return r
    return comp


def census_verdict(rec):
    """What one census record says, in the form expected.json keeps."""
    if rec is None:
        return {'planner': 'missing'}
    if rec.get('error'):
        return {'planner': 'error', 'error': rec['error']}
    if rec.get('accepted'):
        return {'planner': 'accepted', 'kinds': rec.get('kinds', [])}
    return {'planner': 'refused', 'code': rec.get('code'), 'subject': rec.get('subject'),
            'fact': rec.get('fact'), 'kinds': rec.get('kinds', [])}


def run_census(args):
    cat = catalogue()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(',')) if args.only else None
    comps = [(p, c) for p in cat['packages'] for c in p['components'] if not only or c['id'] in only]
    records = out / 'census.jsonl'
    done = set()
    if records.exists() and args.resume:
        done = {json.loads(l)['id'] for l in open(records) if '"component_done"' in l}
    else:
        records.write_text('')
    optdir = out / 'options'
    optdir.mkdir(exist_ok=True)
    todo = [(p, c) for p, c in comps if c['id'] not in done]
    t0 = time.time()
    # --jobs processes side by side, each on its own share of the components and its own output
    # file (CPU only: the census opens no device), appended to census.jsonl when all are done.
    jobs = max(1, min(args.jobs, len(todo))) if todo else 0
    procs = []
    for j in range(jobs):
        share = todo[j::jobs]
        listing = out / f'list.{j}.txt'
        with open(listing, 'w') as f:
            for p, c in share:
                pkg = pathlib.Path(args.corpus) / f"{c['id']}.tar.gz"
                f.write(f"{c['id']} {pkg} {write_options(cat, c, 'app', optdir)}\n")
        part = out / f'census.{j}.jsonl'
        part.write_text('')
        log = open(out / f'census.{j}.log', 'w')
        procs.append((subprocess.Popen([args.tool, '--list', str(listing), '--out', str(part)],
                                       stdout=log, stderr=subprocess.STDOUT), part, log))
    for proc, part, log in procs:
        try:
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 'timeout'
        log.close()
        if rc != 0:
            print(f'census tool {part.name}: exit {rc}', file=sys.stderr)
        with open(records, 'a') as f:
            f.write(part.read_text())
        part.unlink()
    seconds = time.time() - t0
    if not os.path.exists(args.expected):
        print(f'census recorded in {records} ({len(todo)} components, {seconds:.1f} s); no {args.expected} to check against')
        return 0
    result = check_census(read_census(records), load_json(args.expected), cat,
                          read_census(args.compare_with) if args.compare_with else None,
                          only)
    for line in open(records):
        if '"record":"end"' in line:
            held = json.loads(line).get('open_device_nodes')
            if held:
                result['failures'].append(f'the census held device nodes at its end: {held}')
                result['ok'] = False
    result['seconds'] = round(seconds, 1)
    result['components_run'] = len(todo)
    write_json(out / 'census-result.json', result)
    print_census(result)
    return 0 if result['ok'] else 1


def check_census(recs, expected, cat, other=None, only=None):
    """Every component and view against expected.json: the planner's verdict, and for a refusal
    its code, subject and fact, which must also keep the standard's form; on the stock firmware
    every route with an EV74 kernel stage must be refused with the firmware form."""
    rows = {(r['package'], r['view']): r for r in expected['rows']}
    failures, notes = [], []
    counts = collections.Counter()
    for p, view in component_views(cat):
        row = rows.get((p['name'], view))
        if row is None:
            failures.append(f"{p['name']} {view}: no row in expected.json")
            continue
        exp_comps = {c['id']: c for c in row['components']}
        for c in p['components']:
            if only and c['id'] not in only:
                continue
            got = census_verdict(recs.get((c['id'], view)))
            exp = exp_comps.get(c['id'], {})
            key = f"{p['name']} {view} {c['id']}"
            counts[got['planner']] += 1
            if got['planner'] in ('missing', 'error'):
                failures.append(f"{key}: census {got['planner']}: {got.get('error', '')}")
                continue
            if got['planner'] != exp.get('planner'):
                failures.append(f"{key}: planner {got['planner']}, expected.json says {exp.get('planner')}"
                                + (f" ({got['subject']}: {got['fact']})" if got['planner'] == 'refused' else ''))
                continue
            if got['planner'] == 'refused':
                for field in ('code', 'subject', 'fact'):
                    if got[field] != exp.get(field):
                        failures.append(f"{key}: {field} {got[field]!r}, expected.json says {exp.get(field)!r}")
                for bad in check_sentence(got['code'], got['subject'], got['fact']):
                    failures.append(f'{key}: refusal breaks the standard: {bad}')
                for fr in exp.get('form_refusals', []):
                    for bad in check_sentence(fr.get('code'), fr.get('subject'), fr.get('fact')):
                        failures.append(f"{key}: the refusal of forms {fr.get('forms')} breaks the standard: {bad}")
                    if not set(fr.get('forms', [])) <= set(FORMS) or not fr.get('forms'):
                        failures.append(f"{key}: form_refusals names forms {fr.get('forms')}")
            rec = recs[(c['id'], view)]
            ev74 = any(k.endswith('@EV74') for k in rec.get('kinds', []))
            if ev74 and rec.get('stock_accepted'):
                failures.append(f'{key}: an EV74 kernel stage accepted on the stock firmware')
            if ev74 and not rec.get('stock_accepted') and got['planner'] == 'accepted':
                if 'stock firmware' not in rec.get('stock_fact', '') and 'firmware' not in rec.get('stock_fact', ''):
                    failures.append(f"{key}: stock-firmware refusal does not name the firmware: {rec.get('stock_sentence')}")
            if rec.get('options_not_applied'):
                notes.append(f"{key}: options not applied: {rec['options_not_applied']}")
            if other is not None:
                o = census_verdict(other.get((c['id'], view)))
                if o != got:
                    failures.append(f'{key}: differs from the other census: {o} against {got}')
    pk = collections.Counter()
    for (name, view), row in rows.items():
        pk[(view, row['outcome'])] += 1
    return {'ok': not failures, 'failures': failures, 'notes': notes, 'components': dict(counts),
            'rows': {f'{v} {o}': n for (v, o), n in sorted(pk.items())}}


def print_census(result):
    print(f"census: {'OK' if result['ok'] else 'FAILED'}; components {result['components']}; "
          f"rows {result['rows']}; {result.get('seconds', '?')} s")
    for f in result['failures'][:200]:
        print('  FAIL', f)


def seed(args):
    """expected.json from a census: per package and view, the outcome and per component the
    planner's verdict; hand-written fields of an existing file (forms, notes) are kept."""
    cat = catalogue()
    recs = read_census(args.census)
    old = load_json(args.out) if os.path.exists(args.out) else {'rows': []}
    old_rows = {(r['package'], r['view']): r for r in old.get('rows', [])}
    obstacles = load_json(args.obstacles) if args.obstacles else {}
    disagree = []
    rows = []
    for p, view in component_views(cat):
        comps = []
        for c in p['components']:
            v = census_verdict(recs.get((c['id'], view)))
            if v['planner'] in ('missing', 'error'):
                sys.exit(f"{c['id']} {view}: census {v['planner']} {v.get('error', '')}")
            entry = {'id': c['id'], **v}
            prev_comp = next((x for x in old_rows.get((p['name'], view), {}).get('components', [])
                              if x.get('id') == c['id']), {})
            if v['planner'] == 'refused' and prev_comp.get('form_refusals'):
                entry['form_refusals'] = prev_comp['form_refusals']
            ob = obstacles.get(f"{c['id']} {view}")
            if ob is not None:
                if ob.get('tally') != v['planner']:
                    disagree.append(f"{c['id']} {view}: census {v['planner']}, the earlier census {ob.get('tally')}")
                if v['planner'] == 'refused':
                    entry['obstacles'] = ob.get('units', [])
                    entry['phase'] = ob.get('phase')
                    entry['owners'] = ob.get('owners', [])
            comps.append(entry)
        outcome = 'equal' if all(x['planner'] == 'accepted' for x in comps) else 'not_carried'
        row = {'package': p['name'], 'view': view, 'outcome': outcome, 'components': comps}
        if outcome == 'not_carried' and any('phase' in x for x in comps):
            # the PLAN.md phase after which the earlier census expects the row to flip
            row['flips_at_phase'] = max(x.get('phase') or 0 for x in comps)
        prev = old_rows.get((p['name'], view), {})
        for keep in ('forms', 'notes'):
            if keep in prev:
                row[keep] = prev[keep]
        rows.append(row)
    out = {k: v for k, v in old.items() if k != 'rows'}
    out['rows'] = rows
    write_json(args.out, out)
    c = collections.Counter((r['view'], r['outcome']) for r in rows)
    print(dict(c))
    for d in disagree:
        print('DISAGREES WITH THE EARLIER CENSUS:', d)
    return 0 if not disagree else 1


# ---- comparator (T1/T2) ----------------------------------------------------------------------------------
def read_records(path):
    recs = []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{'):
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    recs.append({'record': 'corrupt', 'line': line[:200]})
    return recs


def flatten(obj, prefix=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f'{prefix}.{k}' if prefix else k))
    elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        out[prefix + '#'] = len(obj)
        for i, x in enumerate(obj):
            out.update(flatten(x, f'{prefix}[{i}]'))
    else:
        out[prefix] = obj
    return out


# Fields of a record that say how the tool worked, not what the backend gave.
TOOL_FIELDS = re.compile(r'(^|\.)(seconds|hashed)$|^(record|form|sc|i|error)$|^first_error$|^values\.')


def index_records(recs):
    """{(form, group-or-scenario, kind, i): record}"""
    idx = {}
    for r in recs:
        kind = r.get('record')
        if kind == 'frame':
            idx[(r['form'], r['sc'], 'frame', r['i'])] = r
        elif kind == 'end':
            idx[(r['form'], r['sc'], 'end', 0)] = r
        elif kind in ('build', 'close'):
            idx[(r['form'], r['group'], kind, 0)] = r
        elif kind in ('plan', 'bench', 'model', 'process'):
            idx[(kind, '', kind, 0)] = r
    return idx


def diffs(a_recs, b_recs, forms):
    """Every field that differs between two processes' records: (form, sc, kind, i, path, a, b)."""
    a, b = index_records(a_recs), index_records(b_recs)
    out = []
    for key in sorted(set(a) | set(b), key=str):
        form, sc, kind, i = key
        if kind in ('process',) or (form not in forms and kind not in ('model',)):
            continue
        ra, rb = a.get(key), b.get(key)
        if ra is None or rb is None:
            out.append((form, sc, kind, i, '<record>', 'present' if ra else 'absent',
                        'present' if rb else 'absent', (ra or rb).get('input', '')))
            continue
        fa, fb = flatten(ra), flatten(rb)
        for path in sorted(set(fa) | set(fb)):
            if TOOL_FIELDS.search(path) or path.startswith('json'):
                continue
            va, vb = fa.get(path, '<absent>'), fb.get(path, '<absent>')
            if va != vb:
                out.append((form, sc, kind, i, path, va, vb, ra.get('input', '')))
    return out


def pattern(path):
    return re.sub(r'\[\d+\]', '[*]', path)


def field_matches(path_pattern, field):
    """A row's field names a record path; [*] stands for any index (paths are compared with their
    indices as [*]) and * elsewhere for any text."""
    return re.fullmatch(re.escape(field).replace(r'\*', '.*'), path_pattern) is not None


def value_matches(spec, value):
    if spec == '*':
        return True
    if isinstance(spec, str) and spec.startswith('re:'):
        return re.fullmatch(spec[3:], str(value)) is not None
    if isinstance(spec, list):
        return any(value_matches(s, value) for s in spec)
    return spec == value


def row_applies(row, view, form, sc, label=''):
    scope = row.get('scope', {})
    return ((not scope.get('views') or view in scope['views']) and
            (not scope.get('inputs') or any(re.fullmatch(x, label) for x in scope['inputs'])) and
            (not scope.get('forms') or form in scope['forms']) and
            (not scope.get('scenarios') or sc in scope['scenarios']))


def relation_holds(rel, g, d):
    if 'offset' in rel:
        return isinstance(g, int) and isinstance(d, int) and g - d == rel['offset']
    if 'rewrite' in rel:
        # GStreamer's value, rewritten by a regular expression, is Dataflow's
        pat, repl = rel['rewrite']
        return isinstance(g, str) and re.fullmatch(pat, g) is not None and re.sub(pat, repl, g) == d
    return value_matches(rel.get('gstreamer', '*'), g) and value_matches(rel.get('dataflow', '*'), d)


def active_rows(differences):
    return [r for r in differences['rows'] if r.get('state') == 'active' and r.get('check') == 'fields']


def explain(diff, rows, view):
    """The id of the row that explains one difference, or None."""
    form, sc, kind, i, path, g, d, label = diff
    p = pattern(path)
    for row in rows:
        if not row_applies(row, view, form, sc, label):
            continue
        if any(field_matches(p, f) for f in row['fields']) and relation_holds(row['relation'], g, d):
            return row['id']
    return None


def stale_rows(g_recs, d_recs, rows, view, forms):
    """Active rows whose fields were compared in their scope and were equal everywhere: the
    difference they document is gone, so the row must go (or the comparator is not looking)."""
    seen = collections.Counter()
    differing = collections.Counter()
    a, b = index_records(g_recs), index_records(d_recs)
    for key in set(a) & set(b):
        form, sc, kind, i = key
        if form not in forms:
            continue
        fa, fb = flatten(a[key]), flatten(b[key])
        for path in set(fa) & set(fb):
            p = pattern(path)
            for row in rows:
                if row.get('must_observe') and row_applies(row, view, form, sc, a[key].get('input', '')) and \
                        any(field_matches(p, f) for f in row['fields']):
                    seen[row['id']] += 1
                    if fa[path] != fb[path]:
                        differing[row['id']] += 1
    return sorted(r for r in seen if differing[r] == 0)


def norm_line(line):
    line = re.sub(r'proc_run_\w+', 'proc_run_X', line)
    line = re.sub(r'pkg_[0-9a-f]+', 'pkg_X', line)
    line = re.sub(r'0x[0-9a-fA-F]+', '0xN', line)
    line = re.sub(r'\d+(\.\d+)?', 'N', line)
    return line.strip()


def text_lines(text):
    return collections.Counter(norm_line(l) for l in (text or '').splitlines() if l.strip())


def stderr_diff(g_text, d_text, floor_lines, differences):
    """Lines one process printed and the other did not (numbers and paths normalised), not in the
    floor and not named by an active stderr row."""
    g, d = text_lines(g_text), text_lines(d_text)
    rows = [r for r in differences['rows'] if r.get('state') == 'active' and r.get('check') == 'stderr']
    out = []
    for side, extra in (('gstreamer', g - d), ('dataflow', d - g)):
        for line in extra:
            if line in floor_lines:
                continue
            if any(r['side'] == side and re.search(r['regex'], line) for r in rows):
                continue
            out.append(f'{side} only: {line}')
    return out


def build_of(recs, form, group):
    for r in recs:
        if r.get('record') == 'build' and r.get('form') == form and r.get('group') == group:
            return r
    return None


def frames_ok(recs, form, stops=None):
    """Whether every scenario of `form` delivered every frame; a scenario in `stops` (a documented
    stop: documented_stops) delivered every frame up to the one both backends refuse."""
    ends = [r for r in recs if r.get('record') == 'end' and r.get('form') == form]
    return ends and all(r.get('frames_ok') == r.get('frames') or (form, r.get('sc')) in (stops or {})
                        for r in ends), ends


def documented_stops(g_recs, d_recs, differences, view, comp_id, forms):
    """{(form, scenario): (frame index, row id)} of the scenarios that end early on both backends
    as an active row of check `stop` says: at the same frame, whose input is in the row's scope,
    each backend with the row's error (GStreamer's pull, Dataflow's push). And the ids of the
    must_observe stop rows in scope of this component whose stop did not happen."""
    rows = [r for r in differences['rows'] if r.get('state') == 'active' and r.get('check') == 'stop']
    a, b = index_records(g_recs), index_records(d_recs)
    found, seen, hit = {}, set(), set()
    for row in rows:
        scope = row.get('scope', {})
        if scope.get('components') and not any(re.fullmatch(x, comp_id) for x in scope['components']):
            continue
        for key in sorted(set(a) & set(b), key=str):
            form, sc, kind, _ = key
            if kind != 'end' or form not in forms:
                continue
            ga, db = a[key], b[key]
            labels = [r.get('input', '') for (f, s, k, _), r in a.items() if (f, s, k) == (form, sc, 'frame')]
            labels += [r.get('input', '') for (f, s, k, _), r in b.items() if (f, s, k) == (form, sc, 'frame')]
            if not any(row_applies(row, view, form, sc, label) for label in labels):
                if not scope.get('inputs') and row_applies(row, view, form, sc):
                    seen.add(row['id'])
                continue
            seen.add(row['id'])
            i = ga.get('frames_ok')
            gf, df = a.get((form, sc, 'frame', i)), b.get((form, sc, 'frame', i))
            if i is None or i != db.get('frames_ok') or i >= ga.get('frames', 0) or gf is None or df is None:
                continue
            if not (row_applies(row, view, form, sc, gf.get('input', '')) and
                    gf.get('input') == df.get('input') and
                    re.fullmatch(row['gstreamer_error'], ga.get('first_error', ''), re.S) and
                    re.fullmatch(row['dataflow_error'], db.get('first_error', ''), re.S)):
                continue
            found[(form, sc)] = (i, row['id'])
            hit.add(row['id'])
    stale = sorted(r['id'] for r in rows if r.get('must_observe') and r['id'] in seen and r['id'] not in hit)
    return found, stale


def form_state(row, expected, form):
    over = row.get('forms', {}).get(form)
    return over or expected.get('forms', {}).get(form, {'state': 'active'})


def judge_component(comp, row, expected, differences, g, d, floor, view, forms, check_stderr=True):
    """One component: g, d, floor are {'records', 'stderr', 'rc'} of the GStreamer, Dataflow and
    second GStreamer process. Returns a dict with 'outcome' and the evidence."""
    out = {'id': comp['id'], 'forms': {}, 'unexplained': [], 'explained': collections.Counter(),
           'floor_paths': [], 'stale_rows': [], 'stderr': []}
    for who, proc in (('gstreamer', g), ('dataflow', d)):
        if proc['rc'] != 0:
            out['outcome'] = 'FAIL'
            out['why'] = f'the {who} process exited {proc["rc"]}: {proc["stderr"][-400:]}'
            return out
        if not any(r.get('record') == 'done' for r in proc['records']):
            out['outcome'] = 'FAIL'
            out['why'] = f'the {who} process wrote no end record'
            return out
    # Nothing of GStreamer in a process on the Dataflow backend (DECISIONS.md 31): at its end it has
    # not initialised GStreamer and holds no GStreamer type (compare.cpp's done record).
    gst = next(r for r in d['records'] if r.get('record') == 'done').get('gstreamer')
    if gst is None or gst.get('initialized') or gst.get('types'):
        out['outcome'] = 'FAIL'
        out['why'] = f'the dataflow process holds GStreamer at its end: {gst}'
        return out
    rows = active_rows(differences)
    floor_paths = set()
    floor_lines = set()
    if floor is not None:
        for f in diffs(g['records'], floor['records'], forms):
            floor_paths.add((f[0], f[1], pattern(f[4])))
        gl, fl = text_lines(g['stderr']), text_lines(floor['stderr'])
        floor_lines = set((gl - fl) + (fl - gl))
    out['floor_paths'] = sorted({f'{a}/{b}/{c}' for a, b, c in floor_paths})
    verdicts = []
    for form in forms:
        state = form_state(row, expected, form)
        first = 'run' if form == 'model_run' else ('plan' if form == 'plan' else
                                                    'bench' if form == 'bench' else 'lock')
        gb, db = build_of(g['records'], form, first), build_of(d['records'], form, first)
        fv = {'state': state.get('state')}
        if gb is None or not gb.get('ok'):
            fv['outcome'] = 'FAIL'
            fv['why'] = 'GStreamer did not build: ' + (gb or {}).get('error', 'no build record')
            out['forms'][form] = fv
            verdicts.append(fv['outcome'])
            continue
        stops, unstopped = documented_stops(g['records'], d['records'], differences, view, comp['id'], [form])
        g_ok, _ = frames_ok(g['records'], form, stops)
        if form not in ('plan', 'bench') and not g_ok:
            fv['outcome'] = 'FAIL'
            fv['why'] = 'GStreamer did not deliver every frame'
            out['forms'][form] = fv
            verdicts.append('FAIL')
            continue
        if db is None or not db.get('ok'):
            err = (db or {}).get('error', 'no build record')
            parts = parse_one_form(err)
            fv['dataflow_error'] = err
            if parts is None:
                fv['outcome'] = 'FAIL'
                fv['why'] = 'Dataflow refused, not in the one form'
            elif state.get('state') == 'pending':
                fv['outcome'] = 'pending'
                fv['refusal'] = {k: parts[k] for k in ('code', 'where', 'subject', 'fact')}
            elif comp.get('planner') == 'refused':
                exp = expected_refusal(comp, form)
                same = all(parts[k] == exp.get(k) for k in ('code', 'subject', 'fact'))
                fv['outcome'] = 'not_carried' if same else 'FAIL'
                fv['refusal'] = {k: parts[k] for k in ('code', 'where', 'subject', 'fact')}
                if not same:
                    fv['why'] = 'refusal differs from expected.json (code, subject or fact)'
            else:
                fv['outcome'] = 'FAIL'
                fv['why'] = 'the planner accepts this component, the build refused it'
            out['forms'][form] = fv
            verdicts.append(fv['outcome'])
            continue
        if comp.get('planner') == 'refused' and form == 'graph':
            fv['outcome'] = 'FAIL'
            fv['why'] = 'expected.json says not carried; Dataflow built it'
            out['forms'][form] = fv
            verdicts.append('FAIL')
            continue
        unexplained = []
        for diff in diffs(g['records'], d['records'], [form]):
            if (diff[0], diff[1], pattern(diff[4])) in floor_paths:
                out['explained']['floor'] += 1
                continue
            rid = explain(diff, rows, view)
            stop = stops.get((diff[0], diff[1]))
            if rid is None and stop is not None and (diff[2] == 'end' or (diff[2] == 'frame' and diff[3] == stop[0])):
                # the frame both backends refuse, each in its way, and what the Run says after it
                rid = stop[1]
            if rid is None and diff[2] == 'close':
                # the close of a Run whose scenario a documented stop ended
                rid = next((row for (form, _), (_, row) in stops.items() if form == diff[0]), None)
            if rid:
                out['explained'][rid] += 1
            else:
                unexplained.append(diff)
        stale = stale_rows(g['records'], d['records'], rows, view, [form]) + unstopped
        d_ok, _ = frames_ok(d['records'], form, stops)
        if unexplained or stale or (form not in ('plan', 'bench') and not d_ok):
            fv['outcome'] = 'FAIL'
            fv['why'] = ('Dataflow did not deliver every frame; ' if not d_ok and form not in ('plan', 'bench') else '') + \
                        (f'{len(unexplained)} unexplained differences; ' if unexplained else '') + \
                        (f'rows no longer observed: {stale}' if stale else '')
            fv['first_differences'] = [list(x) for x in unexplained[:12]]
            out['unexplained'] += [list(x) for x in unexplained[:50]]
            out['stale_rows'] += stale
        elif state.get('state') == 'pending':
            fv['outcome'] = 'FLIP'
            fv['why'] = f"form {form} is equal on Dataflow now: set it active in expected.json"
        else:
            fv['outcome'] = 'equal'
            fv['frames'] = sum(r.get('frames_ok', 0) for r in d['records']
                               if r.get('record') == 'end' and r.get('form') == form)
        out['forms'][form] = fv
        verdicts.append(fv['outcome'])
    if check_stderr:
        out['stderr'] = stderr_diff(g['stderr'], d['stderr'], floor_lines, differences)
        # A component Dataflow builds nothing of: what GStreamer's process printed for the builds
        # and runs it made (their progress lines, a benchmark's summary) has no Dataflow
        # counterpart, so only the lines Dataflow printed and GStreamer did not are compared.
        if not any(r.get('record') == 'build' and r.get('ok') for r in d['records']):
            out['stderr'] = [line for line in out['stderr'] if not line.startswith('gstreamer only: ')]
    # The component's outcome: what its first form (the route's build) says, with every other
    # form either agreeing or pending.
    first_outcome = verdicts[0] if verdicts else 'FAIL'
    if 'FAIL' in verdicts or 'FLIP' in verdicts or out['stderr']:
        out['outcome'] = 'FAIL'
        out['why'] = '; '.join(f"{f}: {v.get('why', v['outcome'])}" for f, v in out['forms'].items()
                               if v['outcome'] in ('FAIL', 'FLIP'))
        if out['stderr']:
            out['why'] += ('; ' if out['why'] else '') + f"stderr: {out['stderr'][:5]}"
    else:
        out['outcome'] = first_outcome
    out['explained'] = dict(out['explained'])
    return out


def judge_row(row, results):
    """A package and view: equal when every component is; not_carried when a refused component
    gave the expected refusal and every other component is equal."""
    outs = [r['outcome'] for r in results]
    if any(o not in ('equal', 'not_carried', 'pending') for o in outs):
        return 'FAIL'
    if row['outcome'] == 'equal':
        return 'equal' if all(o == 'equal' for o in outs) else 'FAIL'
    refused = [r for r, c in zip(results, row['components']) if c.get('planner') == 'refused']
    others = [r for r, c in zip(results, row['components']) if c.get('planner') != 'refused']
    if refused and all(r['outcome'] == 'not_carried' for r in refused) and \
            all(r['outcome'] == 'equal' for r in others):
        return 'not_carried'
    return 'FAIL'


def run_process(args, backend, comp, opts, recfile, kernel_log_baseline, forms):
    if kernel_log_baseline is not None:
        n = kernel_log_lines()
        if n != kernel_log_baseline:
            raise SystemExit(f'STOP: the kernel log has {n} lines, the baseline {kernel_log_baseline}')
    cmd = [args.tool, '--backend', backend, '--package', str(pathlib.Path(args.packages) / f"{comp['id']}.tar.gz"),
           '--opts', opts, '--images', args.images, '--out', str(recfile), '--forms', forms,
           '--frames', str(args.frames), '--timeout-ms', str(args.timeout_ms)] + shlex.split(args.tool_args)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.process_timeout)
        rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = 124, e.stdout or '', (e.stderr or '') + '\nTIMEOUT'
        if isinstance(so, bytes):
            so = so.decode(errors='replace')
        if isinstance(se, bytes):
            se = se.decode(errors='replace')
    recs = read_records(recfile) if os.path.exists(recfile) else []
    return {'rc': rc, 'stderr': (so or '') + (se or ''), 'records': recs, 'wall_s': round(time.time() - t0, 2)}


def kernel_log_lines():
    try:
        return len(subprocess.run(['dmesg'], capture_output=True, text=True).stdout.splitlines())
    except OSError:
        return None


def compare(args):
    cat = catalogue()
    expected = load_json(args.expected)
    differences = load_json(args.differences)
    rows = select_rows(expected, args)
    by_name = {p['name']: p for p in cat['packages']}
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    optdir = out / 'options'
    optdir.mkdir(exist_ok=True)
    forms = args.forms.split(',')
    results = load_json(out / 'results.json') if (out / 'results.json').exists() and args.resume else []
    done = {(r['package'], r['view']) for r in results}
    for row in rows:
        if (row['package'], row['view']) in done:
            continue
        p = by_name[row['package']]
        comps_out = []
        t0 = time.time()
        for comp, ecomp in zip(p['components'], row['components']):
            opts = write_options(cat, comp, row['view'], optdir)
            tag = f"{comp['id']}.{row['view']}"
            # The forms Dataflow carries today run in one group of processes (GStreamer, Dataflow,
            # GStreamer again: the floor) and are judged whole, stderr included; the pending ones
            # run in a group of their own, judged form by form (a refused build prints nothing,
            # so their output is not compared).
            groups = [('', [f for f in forms if form_state(row, expected, f).get('state') != 'pending'], True),
                      ('p', [f for f in forms if form_state(row, expected, f).get('state') == 'pending'], False)]
            res = None
            walls = {}
            for suffix, group_forms, whole in groups:
                if not group_forms:
                    continue
                procs = {}
                for who, backend in (('g', 'gstreamer'), ('d', 'dataflow'), ('f', 'gstreamer')):
                    if who == 'f' and not (args.floor and whole):
                        continue
                    rec = out / 'records' / f'{tag}.{who}{suffix}.jsonl'
                    rec.parent.mkdir(exist_ok=True)
                    procs[who] = run_process(args, backend, comp, opts, rec, args.kernel_log_baseline,
                                             ','.join(group_forms))
                    (out / 'records' / f'{tag}.{who}{suffix}.stderr').write_text(procs[who]['stderr'])
                    walls[who + suffix] = procs[who]['wall_s']
                part = judge_component(ecomp, row, expected, differences, procs['g'], procs['d'],
                                       procs.get('f'), row['view'], group_forms, check_stderr=whole)
                if res is None:
                    res = part
                else:
                    res['forms'].update(part['forms'])
                    if part['outcome'] == 'FAIL':
                        res['outcome'] = 'FAIL'
                        res['why'] = '; '.join(x for x in (res.get('why'), part.get('why')) if x)
                for who in procs:
                    f = out / 'records' / f'{tag}.{who}{suffix}.jsonl'
                    if args.keep and f.exists():
                        with open(f, 'rb') as src, gzip.open(str(f) + '.gz', 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                    f.unlink(missing_ok=True)
            res['wall_s'] = walls
            comps_out.append(res)
        verdict = judge_row(row, comps_out)
        results.append({'package': row['package'], 'view': row['view'], 'expected': row['outcome'],
                        'verdict': verdict, 'seconds': round(time.time() - t0, 1), 'components': comps_out})
        write_json(out / 'results.json', results)
        print(f"{row['package']:40s} {row['view']:8s} expected {row['outcome']:12s} -> {verdict}"
              + ('' if verdict == row['outcome'] else '   ' + '; '.join(c.get('why', '') for c in comps_out)[:300]),
              flush=True)
    write_table(results, out / 'results.md')
    bad = [r for r in results if r['verdict'] != r['expected']]
    print(f'{len(results)} rows, {len(bad)} not as expected')
    return 0 if not bad else 1


def judge(args):
    """Judge again from kept records (records/<id>.<view>.{g,d,f}.jsonl.gz)."""
    expected = load_json(args.expected)
    differences = load_json(args.differences)
    cat = catalogue()
    by_name = {p['name']: p for p in cat['packages']}
    rec = pathlib.Path(args.records)
    forms = args.forms.split(',')
    results = []
    for row in select_rows(expected, args):
        comps_out = []
        for comp, ecomp in zip(by_name[row['package']]['components'], row['components']):
            tag = f"{comp['id']}.{row['view']}"
            res = None
            for suffix, whole in (('', True), ('p', False)):
                procs = {}
                for who in 'gdf':
                    f = rec / f'{tag}.{who}{suffix}.jsonl.gz'
                    if f.exists():
                        se = rec / f'{tag}.{who}{suffix}.stderr'
                        procs[who] = {'rc': 0, 'records': read_records(f),
                                      'stderr': se.read_text() if se.exists() else ''}
                if 'g' not in procs or 'd' not in procs:
                    continue
                group_forms = [f for f in forms if (form_state(row, expected, f).get('state') == 'pending') != whole]
                if suffix == '' and not any(procs['g']['records']) or not group_forms:
                    continue
                part = judge_component(ecomp, row, expected, differences, procs['g'], procs['d'],
                                       procs.get('f'), row['view'], group_forms, check_stderr=whole)
                if res is None:
                    res = part
                else:
                    res['forms'].update(part['forms'])
                    if part['outcome'] == 'FAIL':
                        res['outcome'] = 'FAIL'
                        res['why'] = '; '.join(x for x in (res.get('why'), part.get('why')) if x)
            if res is not None:
                comps_out.append(res)
        if comps_out:
            results.append({'package': row['package'], 'view': row['view'], 'expected': row['outcome'],
                            'verdict': judge_row(row, comps_out), 'components': comps_out})
    write_json(pathlib.Path(args.out), results)
    write_table(results, pathlib.Path(args.out).with_suffix('.md'))
    bad = [r for r in results if r['verdict'] != r['expected']]
    print(f'{len(results)} rows, {len(bad)} not as expected')
    return 0 if not bad else 1


def write_table(results, path):
    lines = ['| package | view | expected | verdict | frames compared | evidence |', '|---|---|---|---|---|---|']
    for r in results:
        frames = sum(f.get('frames', 0) for c in r['components'] for f in c['forms'].values())
        ev = []
        for c in r['components']:
            for form, f in c['forms'].items():
                if f['outcome'] in ('not_carried', 'pending') and f.get('refusal'):
                    ev.append(f"{form} {f['outcome']}: [{f['refusal']['code']}] {f['refusal']['where']}: "
                              f"{f['refusal']['subject']}")
                elif f['outcome'] not in ('equal',):
                    ev.append(f"{form} {f['outcome']}: {f.get('why', '')}")
            if c.get('outcome') == 'FAIL' and c.get('why') and not c['forms']:
                ev.append(c['why'])
            if c.get('floor_paths'):
                ev.append('floor: ' + ', '.join(c['floor_paths'][:4]))
        lines.append(f"| {r['package']} | {r['view']} | {r['expected']} | {r['verdict']} | {frames} | "
                     f"{'; '.join(ev)[:600].replace('|', '/')} |")
    pathlib.Path(path).write_text('\n'.join(lines) + '\n')


# ---- the catalogue's own scripts (T3) ------------------------------------------------------------------
SCRIPT_NAMES = ('performance', 'accuracy')
BACKENDS = ('gstreamer', 'dataflow')


def package_id(p):
    return f"{p['model']}__{p['variant']}"


def look_through(directory):
    """An extracted dataset, looking through a single top-level directory as the registry's
    run_model_performance.sh and run_model_accuracy.sh do."""
    d = pathlib.Path(directory)
    entries = sorted(d.iterdir()) if d.is_dir() else []
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else d


def script_arguments(spec, p, script, args, report_dir):
    """The script's own arguments, as the registry's run scripts give them: one --mpk (a
    multi-component model and every accuracy.py: <component>=<path> each), the model's first
    accuracy dataset as <source id>=<directory>, --resources where the script requires it, --output,
    and for accuracy.py --limit."""
    m = spec['models'][p['model']]
    named = script == 'accuracy' or len(p['components']) > 1
    out = []
    for c in p['components']:
        f = pathlib.Path(args.packages) / f"{c['id']}.tar.gz"
        out += ['--mpk', f"{c['component']}={f}" if named else str(f)]
    if m.get('dataset'):
        out += ['--dataset', f"{m['dataset']}={look_through(pathlib.Path(args.datasets) / m['dataset'])}"]
    if m[script].get('resources'):
        out += ['--resources', str(pathlib.Path(args.resources) / package_id(p))]
    out += ['--output', str(report_dir)]
    if script == 'accuracy':
        out += ['--limit', str(args.limit or spec['limit'])]
    return out


def kernel_log():
    try:
        return subprocess.run(['dmesg'], capture_output=True, text=True).stdout.splitlines()
    except OSError:
        return None


def run_script(args, spec, p, script, backend, run_dir):
    """One script on one backend in a process of its own; everything it did is kept in run_dir:
    run.json (command, exit status, seconds, the kernel log's new lines), launch.json (the
    launcher's record), stdout.txt, stderr.txt and report/ (the script's --output)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ('run.json', 'launch.json', 'stdout.txt', 'stderr.txt'):
        (run_dir / stale).unlink(missing_ok=True)
    shutil.rmtree(run_dir / 'report', ignore_errors=True)
    path = pathlib.Path(args.registry) / 'models' / p['model'] / 'scripts' / f'{script}.py'
    cmd = [args.python, str(LAUNCHER), '--backend', backend, '--record', str(run_dir / 'launch.json'), '--',
           str(path)] + script_arguments(spec, p, script, args, run_dir / 'report')
    env = dict(os.environ)
    # PYTHONPATH as the registry's run scripts set it: its tools (model_report) and the packages
    # the script's requirements file names; pyneat is the installed one unless --pythonpath adds it.
    env['PYTHONPATH'] = os.pathsep.join(x for x in (str(pathlib.Path(args.registry) / 'tools'), args.pythonpath,
                                                    env.get('PYTHONPATH')) if x)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    before = kernel_log()
    t0 = time.time()
    try:
        proc = subprocess.run(shlex.split(args.run_prefix) + cmd, capture_output=True, text=True,
                              timeout=args.process_timeout, env=env, cwd=run_dir)
        rc, so, se = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        rc = 124
        so, se = (x.decode(errors='replace') if isinstance(x, bytes) else (x or '') for x in (e.stdout, e.stderr))
        se += '\nTIMEOUT'
    seconds = round(time.time() - t0, 1)
    after = kernel_log()
    new = after[len(before):] if before is not None and after is not None else []
    accept = re.compile(args.accept_kernel_line) if args.accept_kernel_line and \
        spec['models'][p['model']][script].get('benchmark', {}).get('include_mla_layer_stats') else None
    accepted = [l for l in new if accept and accept.search(l)]
    other = [l for l in new if not (accept and accept.search(l))]
    (run_dir / 'stdout.txt').write_text(so)
    (run_dir / 'stderr.txt').write_text(se)
    run = {'package': p['name'], 'script': script, 'backend': backend, 'command': cmd, 'exit_status': rc,
           'seconds': seconds, 'kernel_log': {'before': None if before is None else len(before),
                                              'accepted_lines': len(accepted), 'other_lines': other}}
    write_json(run_dir / 'run.json', run)
    return run


def load_script_run(run_dir):
    """What run_script kept, or None when the run is not there."""
    run_dir = pathlib.Path(run_dir)
    if not (run_dir / 'run.json').exists():
        return None
    run = load_json(run_dir / 'run.json')
    launch = load_json(run_dir / 'launch.json') if (run_dir / 'launch.json').exists() else {}
    reports = {}
    if (run_dir / 'report').is_dir():
        for f in sorted((run_dir / 'report').rglob('*.json')):
            try:
                reports[str(f.relative_to(run_dir / 'report'))] = load_json(f)
            except json.JSONDecodeError:
                reports[str(f.relative_to(run_dir / 'report'))] = {'<unreadable>': True}
    text = lambda n: (run_dir / n).read_text() if (run_dir / n).exists() else ''
    return {'rc': run['exit_status'], 'run': run, 'launch': launch, 'reports': reports,
            'stdout': text('stdout.txt'), 'stderr': text('stderr.txt')}


def report_differences(g_reports, d_reports, measured):
    """What differs between two backends' report files: a file or field one side lacks, a stated
    field whose values differ, a measured field filled on one side only. A measured field's two
    values are not compared (they are the measurement)."""
    out = []
    for name in sorted(set(g_reports) | set(d_reports)):
        if name not in g_reports or name not in d_reports:
            out.append([name, '<file>', 'present' if name in g_reports else 'absent',
                        'present' if name in d_reports else 'absent'])
            continue
        fg, fd = flatten(g_reports[name]), flatten(d_reports[name])
        for path in sorted(set(fg) | set(fd)):
            vg, vd = fg.get(path, '<absent>'), fd.get(path, '<absent>')
            if path not in fg or path not in fd:
                out.append([name, path, vg, vd])
            elif any(m.search(path) for m in measured):
                number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
                if not (number(vg) and number(vd)) or (vg == 0) != (vd == 0):
                    out.append([name, path, vg, vd])
            elif vg != vd:
                out.append([name, path, vg, vd])
    return out


def last_line(text):
    lines = [l for l in (text or '').splitlines() if l.strip()]
    return lines[-1][:300] if lines else ''


def judge_script(row, spec, script, g, d, differences):
    """One package's script on both backends against the row of the script's view: equal (both ran,
    every report file and field present on both, every stated field and every accuracy metric
    equal, the measured fields filled alike, the same lines printed) or not_carried (GStreamer ran;
    on Dataflow the script ended with the one error form, as expected.json records it for a refused
    component of the row)."""
    out = {'script': script, 'view': row['view'], 'expected': row['outcome']}
    if g is None or d is None:
        out.update(outcome='FAIL', why='no run of ' + ' and '.join(b for b, x in zip(BACKENDS, (g, d)) if x is None))
        return out
    out['seconds'] = {'gstreamer': g['run'].get('seconds'), 'dataflow': d['run'].get('seconds')}
    out['accepted_kernel_lines'] = {b: x['run'].get('kernel_log', {}).get('accepted_lines', 0)
                                    for b, x in zip(BACKENDS, (g, d))}
    other = [l for x in (g, d) for l in x['run'].get('kernel_log', {}).get('other_lines', [])]
    if other:
        out.update(outcome='FAIL', why=f'the kernel log grew by lines not accepted: {other[:3]}')
        return out
    if g['rc'] != 0 or not g['reports']:
        out.update(outcome='FAIL', why=f"GStreamer: the script exited {g['rc']}"
                   + ('' if g['reports'] else ' and wrote no report')
                   + f": {g['launch'].get('exception') or g['launch'].get('exit_message') or last_line(g['stderr'])}"[:400])
        return out
    error = d['launch'].get('error', '')
    parts = parse_one_form(error)
    if row['outcome'] == 'not_carried':
        if d['rc'] == 0:
            out.update(outcome='FAIL', why='expected.json says not carried; the script ran on Dataflow')
            return out
        if parts is None:
            out.update(outcome='FAIL', why='Dataflow failed, not in the one form: '
                       + (d['launch'].get('exception') or last_line(d['stderr']))[:400])
            return out
        out['refusal'] = {k: parts[k] for k in ('code', 'where', 'subject', 'fact')}
        # performance.py reaches the backend through Model::benchmark, accuracy.py through Model::build
        form = 'bench' if script == 'performance' else 'model'
        refused = [expected_refusal(c, form) for c in row['components'] if c.get('planner') == 'refused']
        same = any(all(parts[k] == c.get(k) for k in ('code', 'subject', 'fact')) for c in refused)
        out.update(outcome='not_carried') if same else \
            out.update(outcome='FAIL', why='refusal differs from expected.json (code, subject or fact)')
        return out
    if d['rc'] != 0:
        out.update(outcome='FAIL', why=('Dataflow refused what the planner accepts: ' if parts else
                                        f"Dataflow: the script exited {d['rc']}: ")
                   + (d['launch'].get('exception') or d['launch'].get('exit_message') or last_line(d['stderr']))[:400])
        return out
    measured = [re.compile(x) for x in spec['measured_fields']]
    diff = report_differences(g['reports'], d['reports'], measured)
    lines = stderr_diff(g['stdout'] + g['stderr'], d['stdout'] + d['stderr'], set(), differences)
    out['report_files'] = sorted(g['reports'])
    out['fields'] = sum(len(flatten(r)) for r in g['reports'].values())
    if diff or lines:
        out.update(outcome='FAIL', why='; '.join(x for x in (
            f'{len(diff)} report fields differ: ' + ', '.join(f'{n}:{p}' for n, p, _, _ in diff[:4]) if diff else '',
            f'{len(lines)} printed lines differ: {lines[:3]}' if lines else '') if x),
            report_differences=diff[:40], printed_differences=lines[:20])
        return out
    out['outcome'] = 'equal'
    return out


def script_metrics(run, script):
    """The numbers a report states: accuracy metrics per dataset; latency and throughput per
    component (a single-component report states them at the top)."""
    rep = (run or {}).get('reports', {}).get('model-report.json')
    if not rep:
        return {}
    if script == 'accuracy':
        return {name: {**d.get('metrics', {}), 'evaluated_samples': d.get('evaluated_samples')}
                for name, d in rep.get('datasets', {}).items()}
    if 'metrics' in rep:
        return {'model': rep['metrics']}
    return {name: c.get('metrics', {}) for name, c in rep.get('components', {}).items() if c.get('metrics')}


def scripts_tier(args):
    for name in ('registry', 'packages', 'datasets', 'resources', 'pythonpath', 'out'):  # a run's cwd is its own directory
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    cat = catalogue()
    expected = load_json(args.expected)
    differences = load_json(args.differences)
    spec = load_json(args.scripts)
    rows = {(r['package'], r['view']): r for r in expected['rows']}
    names = set(args.names.split(',')) if args.names else None
    wanted = [s for s in SCRIPT_NAMES if args.script in (s, 'both')]
    backends = [b for b in BACKENDS if b in args.backends.split(',')]
    out = pathlib.Path(args.out)
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    results = []
    for p in cat['packages']:
        if names is not None and p['name'] not in names:
            continue
        if p['model'] not in spec['models']:
            continue
        carried = any(rows[(p['name'], v)]['outcome'] == 'equal' for v in ('app', 'default'))
        if names is None and not (carried or args.all):
            continue
        for script in wanted:
            if script not in spec['models'][p['model']]:
                continue
            view = 'app' if script == 'performance' else spec['models'][p['model']][script].get('view', 'default')
            row = rows[(p['name'], view)]
            runs = {}
            # the order of the two backends alternates from package to package
            order = backends if len(results) % 2 == 0 or not args.alternate else backends[::-1]
            for backend in order:
                run_dir = out / 'runs' / f'{package_id(p)}.{script}.{backend}'
                if not args.judge_only and not (args.resume and (run_dir / 'run.json').exists()):
                    run = run_script(args, spec, p, script, backend, run_dir)
                    print(f"{p['name']:36s} {script:11s} {backend:9s} exit {run['exit_status']} "
                          f"{run['seconds']} s kernel lines accepted {run['kernel_log']['accepted_lines']} "
                          f"other {len(run['kernel_log']['other_lines'])}", flush=True)
                    if run['kernel_log']['other_lines']:
                        print('STOP: the kernel log grew: ' + ' | '.join(run['kernel_log']['other_lines'][:5]))
                        return 22
            for backend in BACKENDS:
                runs[backend] = load_script_run(out / 'runs' / f'{package_id(p)}.{script}.{backend}')
            if runs['gstreamer'] is None and runs['dataflow'] is None:
                continue
            res = judge_script(row, spec, script, runs['gstreamer'], runs['dataflow'], differences)
            res = {'package': p['name'], **res, 'verdict': res.pop('outcome'),
                   'metrics': {b: script_metrics(runs[b], script) for b in BACKENDS}}
            results.append(res)
    if all(b in backends for b in BACKENDS) or args.judge_only:
        write_json(out / 'results.json', results)
        write_scripts_table(results, out / 'results.md')
        for r in results:
            print(f"{r['package']:36s} {r['script']:11s} {r['view']:8s} expected {r['expected']:12s} -> {r['verdict']}"
                  + ('' if r['verdict'] == r['expected'] else '   ' + r.get('why', '')[:300]), flush=True)
        bad = [r for r in results if r['verdict'] != r['expected']]
        print(f'{len(results)} package scripts, {len(bad)} not as expected')
        return 0 if not bad else 1
    return 0


def fmt(v):
    return f'{v:.4g}' if isinstance(v, float) else str(v)


def write_scripts_table(results, path):
    lines = ['| package | script | view | expected | verdict | GStreamer | Dataflow | evidence |',
             '|---|---|---|---|---|---|---|---|']
    for r in results:
        cells = []
        for b in BACKENDS:
            m = r.get('metrics', {}).get(b, {})
            if r['script'] == 'accuracy':
                cells.append('; '.join(', '.join(f'{k} {fmt(v)}' for k, v in d.items()) for d in m.values()))
            else:
                cells.append('; '.join(f"{c}: p50 {fmt(d.get('latency_p50_ms', '-'))} ms, "
                                       f"{fmt(d.get('throughput_fps', '-'))} /s" for c, d in m.items()))
        ev = r.get('why', '')
        if r.get('refusal'):
            ev = f"[{r['refusal']['code']}] {r['refusal']['where']}: {r['refusal']['subject']}: {r['refusal']['fact']}"
        elif r['verdict'] == 'equal':
            ev = f"{r.get('fields', 0)} report fields in {len(r.get('report_files', []))} files"
        lines.append(f"| {r['package']} | {r['script']} | {r['view']} | {r['expected']} | {r['verdict']} | "
                     f"{cells[0]} | {cells[1]} | {ev[:500].replace('|', '/')} |")
    pathlib.Path(path).write_text('\n'.join(lines) + '\n')


def speed_rows(results):
    """Per package and component of the performance.py reports: latency and throughput on both
    backends and Dataflow's change against GStreamer (latency: + is slower; throughput: - is slower)."""
    out = []
    for r in results:
        if r['script'] != 'performance':
            continue
        g, d = r['metrics'].get('gstreamer', {}), r['metrics'].get('dataflow', {})
        for comp in sorted(set(g) & set(d)):
            row = {'package': r['package'], 'component': comp, 'verdict': r['verdict']}
            for k in ('latency_avg_ms', 'latency_p50_ms', 'latency_p99_ms', 'throughput_fps'):
                a, b = g[comp].get(k), d[comp].get(k)
                row[k] = {'gstreamer': a, 'dataflow': b,
                          'change_pct': round((b - a) / a * 100, 2) if a and b is not None else None}
            row['slower'] = [k for k in ('latency_p50_ms', 'throughput_fps')
                             if row[k]['change_pct'] is not None and
                             (row[k]['change_pct'] > 0) == (k != 'throughput_fps') and row[k]['change_pct'] != 0]
            out.append(row)
    return out


def speed(args):
    rows = speed_rows(load_json(args.results))
    lines = ['| package | component | latency p50 ms G | D | change | latency p99 ms G | D | throughput /s G | D | change | slower on Dataflow |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        l, q, f = r['latency_p50_ms'], r['latency_p99_ms'], r['throughput_fps']
        pct = lambda x: '-' if x['change_pct'] is None else f"{x['change_pct']:+.2f} %"
        lines.append(f"| {r['package']} | {r['component']} | {fmt(l['gstreamer'])} | {fmt(l['dataflow'])} | {pct(l)} | "
                     f"{fmt(q['gstreamer'])} | {fmt(q['dataflow'])} | {fmt(f['gstreamer'])} | {fmt(f['dataflow'])} | "
                     f"{pct(f)} | {', '.join(x.replace('_ms', '').replace('_fps', '') for x in r['slower']) or 'no'} |")
    text = '\n'.join(lines) + '\n'
    if args.out:
        pathlib.Path(args.out).write_text(text)
        write_json(pathlib.Path(args.out).with_suffix('.json'), rows)
    else:
        print(text, end='')
    slower = [r for r in rows if r['slower']]
    print(f"{len(rows)} components of {len({r['package'] for r in rows})} packages; slower on Dataflow in "
          f"latency p50: {sum('latency_p50_ms' in r['slower'] for r in rows)}, in throughput: "
          f"{sum('throughput_fps' in r['slower'] for r in rows)} ({len(slower)} components in either)")
    return 0



def list_rows(args):
    for r in select_rows(load_json(args.expected), args):
        print(r['package'], r['view'], r['outcome'], ' '.join(c['id'] for c in r['components']))
    return 0


# ---- selftest ------------------------------------------------------------------------------------------
def selftest(_args):
    """The judge's rules on synthetic records."""
    failures = []

    def check(cond, what):
        if not cond:
            failures.append(what)

    good = ("[misconfig.pipeline_shape] Graph::build(inputs): stage 1 'MLA_0' (model on the MLA) is not "
            "carried by the Dataflow backend: it writes 2 output buffers (2 tensors): an MLA stage writes "
            "one output buffer.\nHint: To run this program on the GStreamer backend, call "
            "simaai::neat::set_backend(simaai::neat::RunBackend::GStreamer) (Python: "
            "pyneat.set_backend(pyneat.RunBackend.GStreamer)) before the program constructs its first Model "
            "or builds its first Graph.")
    p = parse_one_form(good)
    check(p and p['code'] == 'misconfig.pipeline_shape' and p['where'] == 'Graph::build(inputs)', 'one form: parts')
    check(p and p['subject'] == "stage 1 'MLA_0' (model on the MLA)", 'one form: subject')
    check(p and p['fact'].endswith('one output buffer'), 'one form: fact')
    route = good.replace("stage 1 'MLA_0' (model on the MLA)",
                         'the model route (3 stages: quant@EV74, mla, dequant@A65)')
    check(parse_one_form(route)['subject'].startswith('the model route (3 stages'), 'one form: subject with a colon')
    check(parse_one_form(good.replace('set_backend', 'set_mode')) is None, 'one form: the hint must name the call')
    check(parse_one_form('[misconfig.pipeline_shape] Graph::build: node Preproc is not supported') is None,
          'one form: other text')
    check(check_sentence('misconfig.pipeline_shape', 's', 'it is not supported yet'), 'standard: "yet"')
    check(not check_sentence('misconfig.pipeline_shape', 's', 'it runs 6 kernel invocations'), 'standard: clean fact')

    def proc(frames, extra_sample=None, extra_tensor=None, build_error=None, stderr=''):
        recs = [{'record': 'process'}]
        if build_error:
            recs.append({'record': 'build', 'form': 'graph', 'group': 'lock', 'ok': False, 'error': build_error})
        else:
            recs.append({'record': 'build', 'form': 'graph', 'group': 'lock', 'ok': True, 'seconds': 1})
            for i in range(frames):
                s = {'input_seq': i + 1, 'caps_string': 'x'}
                s.update(extra_sample(i) if extra_sample else {})
                t = {'xxh64': f'{i:016x}', 'stamp': {'seq': 0}}
                t.update(extra_tensor(i) if extra_tensor else {})
                recs.append({'record': 'frame', 'form': 'graph', 'sc': 'lock', 'i': i, 'push': 'ok',
                             'pull': 'Ok', 'sample': s, 'tensors': [t]})
            recs.append({'record': 'end', 'form': 'graph', 'sc': 'lock', 'frames': frames, 'frames_ok': frames})
        recs.append({'record': 'done', 'gstreamer': {'initialized': False, 'types': []}})
        return {'rc': 0, 'records': recs, 'stderr': stderr}

    differences = {'rows': [
        {'id': 'seq', 'state': 'active', 'check': 'fields', 'must_observe': True,
         'fields': ['sample.input_seq'], 'relation': {'offset': 1}},
        {'id': 'warn', 'state': 'active', 'check': 'stderr', 'side': 'dataflow', 'regex': r'^\[WARN\] ok'}]}
    expected = {'forms': {'graph': {'state': 'active'}}}
    row = {'outcome': 'equal'}
    comp = {'id': 'c', 'planner': 'accepted'}
    g = proc(3)
    d = proc(3, extra_sample=lambda i: {'input_seq': i})
    r = judge_component(comp, row, expected, differences, g, d, g, 'app', ['graph'])
    check(r['outcome'] == 'equal', f'judge: offset row explains input_seq ({r})')
    r = judge_component(comp, row, expected, differences, g, g, g, 'app', ['graph'])
    check(r['outcome'] == 'FAIL' and 'seq' in r['stale_rows'], 'judge: a row whose difference is gone is stale')
    d2 = proc(3, extra_sample=lambda i: {'input_seq': i}, extra_tensor=lambda i: {'xxh64': 'ff'})
    r = judge_component(comp, row, expected, differences, g, d2, g, 'app', ['graph'])
    check(r['outcome'] == 'FAIL' and r['unexplained'], 'judge: a byte difference fails')
    f = proc(3, extra_tensor=lambda i: {'xxh64': 'ee'})
    r = judge_component(comp, row, expected, differences, g, d2, f, 'app', ['graph'])
    check(r['outcome'] == 'equal' and r['floor_paths'], 'judge: a path GStreamer varies on is the floor')
    d3 = proc(3, extra_sample=lambda i: {'input_seq': i}, stderr='[WARN] ok 12\n[WARN] other\n')
    r = judge_component(comp, row, expected, differences, g, d3, g, 'app', ['graph'])
    check(r['outcome'] == 'FAIL' and r['stderr'] == ['dataflow only: [WARN] other'], f'judge: stderr rows ({r["stderr"]})')
    comp_r = {'id': 'c', 'planner': 'refused', 'code': 'misconfig.pipeline_shape',
              'subject': "stage 1 'MLA_0' (model on the MLA)",
              'fact': p['fact']}
    r = judge_component(comp_r, {'outcome': 'not_carried'}, expected, differences, g, proc(0, build_error=good),
                        g, 'app', ['graph'])
    check(r['outcome'] == 'not_carried', 'judge: the expected refusal is not_carried')
    g_ran = dict(g, stderr='[1/4] Initializing runtime...\nNEAT Benchmark\n')
    r = judge_component(comp_r, {'outcome': 'not_carried'}, expected, differences, g_ran,
                        proc(0, build_error=good, stderr='[WARN] other\n'), g_ran, 'app', ['graph'])
    check(r['outcome'] == 'FAIL' and r['stderr'] == ['dataflow only: [WARN] other'],
          f'judge: a refused component compares what Dataflow printed only ({r["stderr"]})')
    r = judge_component(comp_r, {'outcome': 'not_carried'}, expected, differences, g,
                        proc(0, build_error=good.replace('2 output', '3 output')), g, 'app', ['graph'])
    check(r['outcome'] == 'FAIL', 'judge: another fact fails')
    r = judge_component(comp_r, {'outcome': 'not_carried'}, expected, differences, g,
                        proc(0, build_error='[misconfig.pipeline_shape] Graph::build: no'), g, 'app', ['graph'])
    check(r['outcome'] == 'FAIL', 'judge: a refusal out of the form fails')
    # A form that refuses at another point than the planner states its own refusal.
    early = good.replace("stage 1 'MLA_0' (model on the MLA)", 'model m').replace(
        p['fact'], 'it reads 4 inputs').replace('Graph::build(inputs)', 'Model::build')
    comp_f = dict(comp_r, form_refusals=[{'forms': ['model', 'bench'], 'code': 'misconfig.pipeline_shape',
                                          'subject': 'model m', 'fact': 'it reads 4 inputs'}])
    as_model = lambda pr: {**pr, 'records': [dict(x, form='model') if 'form' in x else x for x in pr['records']]}
    exp_m = {'forms': {'graph': {'state': 'active'}, 'model': {'state': 'active'}}}
    r = judge_component(comp_f, {'outcome': 'not_carried'}, exp_m, differences, as_model(g),
                        as_model(proc(0, build_error=early)), None, 'app', ['model'])
    check(r['outcome'] == 'not_carried', f'judge: a form with its own expected refusal ({r})')
    r = judge_component(comp_f, {'outcome': 'not_carried'}, exp_m, differences, g, proc(0, build_error=early),
                        None, 'app', ['graph'])
    check(r['outcome'] == 'FAIL', "judge: a form's own refusal does not stand for the other forms")
    r = judge_component(comp_f, {'outcome': 'not_carried'}, exp_m, differences, as_model(g),
                        as_model(proc(0, build_error=good)), None, 'app', ['model'])
    check(r['outcome'] == 'FAIL', "judge: the planner's refusal does not stand for a form that states its own")
    pend = {'forms': {'graph': {'state': 'pending', 'until': 'S3'}}}
    r = judge_component(comp, row, pend, differences, g, proc(0, build_error=good), g, 'app', ['graph'])
    check(r['outcome'] == 'pending', 'judge: a pending form refused in the one form')
    r = judge_component(comp, row, pend, differences, g, d, g, 'app', ['graph'])
    check(r['outcome'] == 'FAIL' and r['forms']['graph']['outcome'] == 'FLIP', 'judge: a pending form that passes must flip')
    # Nothing of GStreamer in the Dataflow process; GStreamer's own holds what it holds.
    def holding(p, initialized, types):
        p = dict(p, records=[dict(r) for r in p['records']])
        p['records'][-1]['gstreamer'] = {'initialized': initialized, 'types': types}
        return p
    r = judge_component(comp, row, expected, differences, holding(g, True, ['GstObject']), d, g, 'app', ['graph'])
    check(r['outcome'] == 'equal', 'judge: the GStreamer process may hold GStreamer')
    r = judge_component(comp, row, expected, differences, g, holding(d, True, []), g, 'app', ['graph'])
    check(r['outcome'] == 'FAIL' and 'holds GStreamer' in r['why'], 'judge: Dataflow initialised GStreamer')
    r = judge_component(comp, row, expected, differences, g, holding(d, False, ['GstFraction']), g, 'app',
                        ['graph'])
    check(r['outcome'] == 'FAIL' and 'GstFraction' in r['why'], 'judge: Dataflow made a GStreamer type')
    r = judge_component(comp, row, expected, differences, g,
                        dict(d, records=[r for r in d['records'] if r.get('record') != 'done'] + [{'record': 'done'}]),
                        g, 'app', ['graph'])
    check(r['outcome'] == 'FAIL', 'judge: an end record without the footprint fails')
    # A documented stop: both backends end a scenario at the same frame, each with the row's error.
    def stopped(error, push, pull, state, at=2, label='img0@854x480'):
        p = proc(3)
        recs = [r for r in p['records'] if r.get('record') != 'done']
        for i in range(at):
            recs.append({'record': 'frame', 'form': 'graph', 'sc': 'sizes', 'i': i, 'input': 'img0@1280x720',
                         'push': 'ok', 'pull': 'Ok', 'sample': {'input_seq': 4 + i, 'caps_string': 'x'},
                         'tensors': [{'xxh64': 'aa', 'stamp': {'seq': 0}}]})
        recs.append({'record': 'frame', 'form': 'graph', 'sc': 'sizes', 'i': at, 'input': label,
                     'push': push, 'pull': pull, 'error': error})
        recs.append({'record': 'end', 'form': 'graph', 'sc': 'sizes', 'frames': 16, 'frames_ok': at,
                     'first_error': error, 'state': state})
        recs.append({'record': 'done', 'gstreamer': {'initialized': False, 'types': []}})
        return {'rc': 0, 'records': recs, 'stderr': ''}
    stop_row = {'id': 'stop', 'state': 'active', 'check': 'stop', 'must_observe': True,
                'scope': {'forms': ['graph'], 'scenarios': ['sizes'], 'inputs': [r'img\d@854x480'],
                          'components': [r'.*_obb__.*']},
                'gstreamer_error': r'.*equal authoritative affine scales.*',
                'dataflow_error': r'.*this frame is not carried by the Dataflow backend: .*'}
    with_stop = {'rows': [stop_row]}
    obb = {'id': 'm_obb__w8a8__model', 'planner': 'accepted'}
    gs = stopped('Error: oriented boxes require equal authoritative affine scales', 'ok', 'Error', 'dead')
    ds = stopped('[misconfig.input_shape] Run::push: this frame is not carried by the Dataflow backend: x.',
                 'threw', '', 'running')
    r = judge_component(obb, row, expected, with_stop, gs, ds, gs, 'app', ['graph'])
    check(r['outcome'] == 'equal' and r['explained'].get('stop'), f'judge: a documented stop ({r})')
    r = judge_component(comp, row, expected, with_stop, gs, ds, gs, 'app', ['graph'])
    check(r['outcome'] == 'FAIL', 'judge: a stop row out of its components does not excuse a short scenario')
    r = judge_component(obb, row, expected, with_stop, gs, stopped('other', 'threw', '', 'running'), gs, 'app',
                        ['graph'])
    check(r['outcome'] == 'FAIL', "judge: a stop with another error is not the row's")
    r = judge_component(obb, row, expected, with_stop, gs, stopped(ds['records'][-2]['first_error'], 'threw', '',
                                                                   'running', at=1), gs, 'app', ['graph'])
    check(r['outcome'] == 'FAIL', 'judge: the backends must stop at the same frame')
    full = proc(3)
    full['records'][-1:-1] = [{'record': 'frame', 'form': 'graph', 'sc': 'sizes', 'i': 0, 'input': 'img0@854x480',
                               'push': 'ok', 'pull': 'Ok', 'sample': {'input_seq': 4, 'caps_string': 'x'},
                               'tensors': [{'xxh64': 'aa', 'stamp': {'seq': 0}}]},
                              {'record': 'end', 'form': 'graph', 'sc': 'sizes', 'frames': 1, 'frames_ok': 1}]
    r = judge_component(obb, row, expected, with_stop, full, full, full, 'app', ['graph'])
    check(r['outcome'] == 'FAIL' and 'stop' in r['stale_rows'], 'judge: a stop that no longer happens is stale')
    check(judge_row({'outcome': 'equal', 'components': [comp]}, [{'outcome': 'equal'}]) == 'equal', 'row: equal')
    check(judge_row({'outcome': 'not_carried', 'components': [comp_r, comp]},
                    [{'outcome': 'not_carried'}, {'outcome': 'equal'}]) == 'not_carried', 'row: not_carried')
    # T3: a script's two runs against the row of its view.
    spec = {'measured_fields': [r'(^|\.)latency(_[a-z0-9]+)*_(ms|us)$', r'(^|\.)(fps|throughput_fps)$',
                                r'(^|\.)energy_joules$', r'(^|\.)details\.inference_seconds$']}

    def srun(rc=0, report=None, error=None, out='', accepted=0, other=()):
        launch = {'exit_status': rc}
        if error:
            launch.update(error=error, exception='NeatError: ' + error)
        return {'rc': rc, 'launch': launch, 'reports': {'model-report.json': report} if report else {},
                'stdout': out, 'stderr': '',
                'run': {'seconds': 1.0, 'kernel_log': {'accepted_lines': accepted, 'other_lines': list(other)}}}

    perf = lambda lat, fps, joules=0.0, layers=2: {
        'stage': 'performance', 'metrics': {'latency_p50_ms': lat, 'throughput_fps': fps}, 'energy_joules': joules,
        'mla_layer_stats': {'status': 'ok', 'layers': [{'name': f'l{i}', 'latency_us': lat} for i in range(layers)]}}
    acc = lambda top1, s=1.5: {'stage': 'accuracy', 'datasets': {'d': {'metrics': {'top1_accuracy': top1},
                                                                         'details': {'inference_seconds': s}}}}
    row_e = {'view': 'app', 'outcome': 'equal', 'components': [comp]}
    row_n = {'view': 'default', 'outcome': 'not_carried', 'components': [comp_r]}
    no_rows = {'rows': []}
    js = lambda row, script, g, d: judge_script(row, spec, script, g, d, no_rows)
    r = js(row_e, 'performance', srun(report=perf(3.3, 746.8), accepted=1), srun(report=perf(2.3, 742.2), accepted=1))
    check(r['outcome'] == 'equal' and r['accepted_kernel_lines'] == {'gstreamer': 1, 'dataflow': 1},
          f'scripts: measured fields differ in value only ({r})')
    r = js(row_e, 'performance', srun(report=perf(3.3, 746.8)), srun(report=perf(2.3, 0.0)))
    check(r['outcome'] == 'FAIL', 'scripts: a measured field filled on one backend only fails')
    r = js(row_e, 'performance', srun(report=perf(3.3, 746.8)), srun(report=perf(2.3, 742.2, layers=1)))
    check(r['outcome'] == 'FAIL', 'scripts: a report field one backend lacks fails')
    r = js(row_e, 'accuracy', srun(report=acc(0.81), out='{"top1": 0.81}'), srun(report=acc(0.81, 1.2), out='{"top1": 0.81}'))
    check(r['outcome'] == 'equal', f'scripts: equal accuracy metrics ({r})')
    r = js(row_e, 'accuracy', srun(report=acc(0.81)), srun(report=acc(0.80)))
    check(r['outcome'] == 'FAIL' and r['report_differences'], 'scripts: an accuracy metric that differs fails')
    r = js(row_e, 'accuracy', srun(report=acc(0.81)), srun(rc=1, error=good))
    check(r['outcome'] == 'FAIL' and 'planner accepts' in r['why'], 'scripts: a refusal of a carried row fails')
    r = js(row_e, 'accuracy', srun(rc=1), srun(report=acc(0.81)))
    check(r['outcome'] == 'FAIL' and r['why'].startswith('GStreamer'), 'scripts: the script must run on GStreamer')
    r = js(row_e, 'performance', srun(report=perf(3.3, 746.8), other=['[1.0] mla: fault']), srun(report=perf(2.3, 742.2)))
    check(r['outcome'] == 'FAIL' and 'kernel log' in r['why'], 'scripts: a kernel-log line not accepted fails')
    r = js(row_n, 'accuracy', srun(report=acc(0.81)), srun(rc=1, error=good))
    check(r['outcome'] == 'not_carried' and r['refusal']['code'] == 'misconfig.pipeline_shape',
          f'scripts: the expected refusal is not_carried ({r})')
    r = js(row_n, 'accuracy', srun(report=acc(0.81)), srun(rc=1, error=good.replace('2 output', '3 output')))
    check(r['outcome'] == 'FAIL', 'scripts: another fact fails')
    r = js(row_n, 'accuracy', srun(report=acc(0.81)), srun(rc=1, error='RuntimeError: no'))
    check(r['outcome'] == 'FAIL' and 'one form' in r['why'], 'scripts: a failure out of the form fails')
    row_f = {'view': 'default', 'outcome': 'not_carried', 'components': [comp_f]}
    r = js(row_f, 'accuracy', srun(report=acc(0.81)), srun(rc=1, error=early))
    check(r['outcome'] == 'not_carried', f"scripts: accuracy.py meets the model form's refusal ({r})")
    r = js(row_f, 'accuracy', srun(report=acc(0.81)), srun(rc=1, error=good))
    check(r['outcome'] == 'FAIL', "scripts: the planner's refusal where the model form states its own fails")
    r = js(row_n, 'accuracy', srun(report=acc(0.81)), srun(report=acc(0.81)))
    check(r['outcome'] == 'FAIL', 'scripts: a row expected.json says is not carried must flip when it runs')
    sp = speed_rows([{'package': 'p', 'script': 'performance', 'verdict': 'equal', 'metrics': {
        'gstreamer': {'model': {'latency_p50_ms': 2.0, 'throughput_fps': 100.0}},
        'dataflow': {'model': {'latency_p50_ms': 1.0, 'throughput_fps': 99.0}}}}])
    check(len(sp) == 1 and sp[0]['slower'] == ['throughput_fps'] and sp[0]['latency_p50_ms']['change_pct'] == -50.0,
          f'speed: lower throughput is slower, lower latency is not ({sp})')
    for f in failures:
        print('selftest FAIL:', f)
    print(f"selftest: {'OK' if not failures else 'FAILED'}")
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def selection(s):
        s.add_argument('--expected', default=str(EXPECTED))
        s.add_argument('--names', help='package names ("yolov8s w8a8"), comma separated')
        s.add_argument('--view', choices=('app', 'default', 'both'), default='both')
        s.add_argument('--outcome', choices=('equal', 'not_carried', 'all'), default='all')
        s.add_argument('--tier')

    s = sub.add_parser('census')
    s.add_argument('--tool', required=True)
    s.add_argument('--corpus', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--expected', default=str(EXPECTED))
    s.add_argument('--only')
    s.add_argument('--jobs', type=int, default=4, help='census processes side by side')
    s.add_argument('--timeout', type=int, default=3600, help='seconds for one process')
    s.add_argument('--resume', action='store_true')
    s.add_argument('--compare-with', help='another census.jsonl that must give the same verdicts')
    s.set_defaults(fn=run_census)

    s = sub.add_parser('census-check')
    s.add_argument('--census', required=True)
    s.add_argument('--expected', default=str(EXPECTED))
    s.add_argument('--compare-with')
    s.set_defaults(fn=lambda a: (lambda r: (print_census(r), 0 if r['ok'] else 1)[1])(
        check_census(read_census(a.census), load_json(a.expected), catalogue(),
                     read_census(a.compare_with) if a.compare_with else None)))

    s = sub.add_parser('compare')
    selection(s)
    s.add_argument('--tool', required=True)
    s.add_argument('--packages', required=True, help='directory of <component id>.tar.gz')
    s.add_argument('--images', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--differences', default=str(DIFFERENCES))
    s.add_argument('--forms', default='graph')
    s.add_argument('--frames', type=int, default=200)
    s.add_argument('--timeout-ms', type=int, default=20000)
    s.add_argument('--process-timeout', type=int, default=900)
    s.add_argument('--floor', action='store_true', help='a second GStreamer process: the determinism floor')
    s.add_argument('--tool-args', default='', help='more arguments for the comparator (--idle-pull, --hash-bench)')
    s.add_argument('--keep', action='store_true', help='keep the records (gzip)')
    s.add_argument('--resume', action='store_true')
    s.add_argument('--kernel-log-baseline', type=int)
    s.set_defaults(fn=compare)

    s = sub.add_parser('judge')
    selection(s)
    s.add_argument('--records', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--differences', default=str(DIFFERENCES))
    s.add_argument('--forms', default='graph')
    s.set_defaults(fn=judge)

    s = sub.add_parser('seed')
    s.add_argument('--census', required=True)
    s.add_argument('--out', default=str(EXPECTED))
    s.add_argument('--obstacles', help='JSON {"<id> <view>": [obstacle codes]}')
    s.set_defaults(fn=seed)

    s = sub.add_parser('rows')
    selection(s)
    s.set_defaults(fn=list_rows)

    s = sub.add_parser('scripts')
    s.add_argument('--expected', default=str(EXPECTED))
    s.add_argument('--differences', default=str(DIFFERENCES))
    s.add_argument('--scripts', default=str(SCRIPTS))
    s.add_argument('--names', help='package names ("yolov8s w8a8"), comma separated; default: every package '
                                   'with an `equal` row')
    s.add_argument('--all', action='store_true', help='every package of the models scripts.json lists')
    s.add_argument('--script', choices=SCRIPT_NAMES + ('both',), default='both')
    s.add_argument('--backends', default=','.join(BACKENDS))
    s.add_argument('--alternate', action='store_true', help='alternate the backends\' order between packages')
    s.add_argument('--registry', help='a checkout of the Models registry (models/<id>/scripts, tools/model_report)')
    s.add_argument('--packages', help='directory of <component id>.tar.gz')
    s.add_argument('--datasets', help='directory of <source id>/, each an extracted dataset archive')
    s.add_argument('--resources', help='directory of <model>__<variant>/, the package\'s supporting files')
    s.add_argument('--out', required=True)
    s.add_argument('--python', default=sys.executable, help='the interpreter that has pyneat and the scripts\' modules')
    s.add_argument('--pythonpath', help='more module directories, after the registry\'s tools')
    s.add_argument('--run-prefix', default='', help='a command put in front of each run (taskset -c 4-7)')
    s.add_argument('--limit', type=int, help='accuracy.py --limit (default: scripts.json)')
    s.add_argument('--process-timeout', type=int, default=1800)
    s.add_argument('--accept-kernel-line', help='a regular expression: new kernel-log lines matching it are '
                   'counted, not a reason to stop, in runs of a script that requests MLA layer stats')
    s.add_argument('--resume', action='store_true', help='keep the runs already in --out')
    s.add_argument('--judge-only', action='store_true', help='run nothing: judge the runs kept in --out')
    s.set_defaults(fn=scripts_tier)

    s = sub.add_parser('speed')
    s.add_argument('--results', required=True, help='results.json of a `scripts` run')
    s.add_argument('--out', help='the table (markdown; the same rows as JSON beside it)')
    s.set_defaults(fn=speed)

    s = sub.add_parser('selftest')
    s.set_defaults(fn=selftest)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
