"""Create named result directories with reproducible run metadata."""

from datetime import datetime
import inspect
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def model_parameters(model, overrides=None):
    """Expand constructor defaults so omitted settings are recorded too."""
    bound = inspect.signature(model).bind_partial(**(overrides or {}))
    bound.apply_defaults()
    return dict(bound.arguments)


def start_run(path, target, method_version, kind, dataset, parameters):
    """Create a fresh run; refuse existing directories to protect previous results."""
    now = datetime.now(ZoneInfo('Asia/Seoul'))
    if path is None:
        path = ROOT / 'data' / 'outputs' / target / method_version / f'{kind}-{now:%Y%m%d-%H%M%S-%f}'
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    metadata = dict(schema_version=1, target=target, method_version=method_version,
                    kind=kind, dataset=dataset, parameters=parameters, command=sys.argv,
                    code_revision=revision, tracked_code_dirty=dirty,
                    started_at=now.isoformat(), status='incomplete')
    (path / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + '\n')
    return path


def finish_run(path, summary=None):
    """Mark a run complete only after its artifacts have been saved."""
    manifest = Path(path) / 'manifest.json'
    metadata = json.loads(manifest.read_text())
    metadata.update(status='complete', finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat())
    if summary is not None:
        metadata['summary'] = summary
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + '\n')
