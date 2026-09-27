"""Fit JAX coupling/solver configurations with independent Rust validation scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import diffrax
import equinox
import jax
import optax
from fit import fit
from predict_extensions import predict


def write(path, value):
    with open(path, 'x') as f:
        json.dump(value, f, allow_nan=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['model', 'configuration', 'graph-json', 'graph', 'data', 'split', 'training', 'scorer', 'output']:
        parser.add_argument('--' + key, required=True)
    a = parser.parse_args()
    paths = {k: Path(getattr(a, k)) for k in ['model', 'configuration', 'graph_json', 'graph', 'data', 'split', 'training', 'scorer']}
    hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()}
    model, configuration, graph, training = [json.loads(paths[k].read_text()) for k in ['model', 'configuration', 'graph_json', 'training']]
    if hashes['model'] != training['model_sha256']:
        raise ValueError('training export belongs to another checkpoint')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    out = Path(a.output); out.mkdir(exist_ok=False)
    scorer = str(paths['scorer'].resolve())
    common = [str(paths[k]) for k in ['graph', 'data', 'split']]
    subprocess.run([scorer, 'plan', *common, str(paths['model']), 'validation', str(out/'validation-plan.json')], check=True)
    plan = json.loads((out/'validation-plan.json').read_text())
    if plan['partition'] != 'validation':
        raise ValueError('fitting requires a validation plan')
    write(out/'manifest.json', {'schema_version': 1, 'source_commit': source, 'source_worktree_dirty': dirty,
        'input_sha256': hashes, 'validation_plan_sha256': hashlib.sha256((out/'validation-plan.json').read_bytes()).hexdigest(),
        'selection': 'minimum Rust validation MSE; earliest epoch breaks ties',
        'jax': jax.__version__, 'diffrax': diffrax.__version__, 'equinox': equinox.__version__, 'optax': optax.__version__,
        'devices': [str(d) for d in jax.devices()], 'precision': 'float64',
        'scope': 'Epoch-zero fit. Only Rust decodes held-out observations. No test scoring or optimizer resume. Extension equations run in JAX; Rust validates lineage and scores predictions.'})
    def score(candidate):
        epoch = candidate['base_model']['epoch']; path = out/f'epoch-{epoch}.json'
        write(path, candidate)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        prediction = predict(candidate, graph, plan, digest)
        predicted = out/f'validation-{epoch}-predictions.json'
        write(predicted, prediction)
        evaluation = out/f'validation-{epoch}'
        subprocess.run([scorer, 'score', *common, str(path), 'validation', str(predicted), str(evaluation)], check=True)
        result = json.loads((evaluation/'selection.json').read_text())
        if result['epoch'] != epoch or result['partition'] != 'validation' or result['checkpoint_sha256'] != digest:
            raise ValueError('unexpected scorer response')
        return result['mse']
    selected, reports = fit(model, graph, training, source, score,
        lambda i, n: print(f'target {i}/{n}', flush=True) if i % 20 == 0 or i == n else None, configuration)
    # Preserve the bytes bound to validation predictions, not just JSON equivalence.
    selected_path = out/f"epoch-{selected['base_model']['epoch']}.json"
    with open(out/'selected.json', 'xb') as f:
        f.write(selected_path.read_bytes())
    write(out/'selection.json', reports)
    print(f"selected epoch {selected['base_model']['epoch']}", flush=True)

if __name__ == '__main__':
    main()
