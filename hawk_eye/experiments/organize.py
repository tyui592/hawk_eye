"""Catalog historical outputs and migrate them without breaking old paths.

Run without arguments for a dry run; --apply moves directories and leaves relative
symlinks. Original file identities and local hashes are checked after each move.
Cloud-only files are not downloaded. Re-running is safe.
"""

import hashlib
import json
import os
from pathlib import Path


from hawk_eye.paths import ROOT
OUTPUTS = ROOT / 'data' / 'outputs'

# Versions keep their historical names; target disambiguates independent components.
RUNS = [
    ('CT-001', 'classical_court_val', 'court_detection/classical-v1/tenniscourt-val',
     'complete', 'TennisCourtDetector val, 2211 images', 'Initial classical court detector.'),
    ('CT-002', 'classical_court_val_v2', 'court_detection/classical-v2/tenniscourt-val',
     'complete', 'TennisCourtDetector val, 2211 images', 'Relative contrast and empty-line penalty.'),
    ('CT-003', 'classical_court_val_v3', 'court_detection/classical-v3/tenniscourt-val',
     'complete', 'TennisCourtDetector val, 2211 images', 'Camera plausibility filter; current court baseline.'),
    ('BT-001a', 'classical_ball_v1_game9', 'ball_tracking/classical-v1/tracknet-game9-dev-gray',
     'superseded', 'TrackNet game9, 1573 frames, development', 'JPEG IMREAD_GRAYSCALE; superseded by BT-002a.'),
    ('BT-001b', 'classical_ball_v1_game10part', 'ball_tracking/classical-v1/tracknet-game10-subset-gray',
     'superseded', 'TrackNet game10 Clips 1,6,7,8,9; 880 frames', 'Partial data download; not the whole holdout.'),
    ('BT-001c', 'classical_ball_v1_game10', 'ball_tracking/classical-v1/tracknet-game10-gray',
     'superseded', 'TrackNet game10, 2144 frames, holdout', 'JPEG IMREAD_GRAYSCALE; superseded by BT-002b.'),
    ('BT-002a', 'classical_ball_v1b_game9', 'ball_tracking/classical-v1b/tracknet-game9-dev-bgr',
     'complete', 'TrackNet game9, 1573 frames, development', 'BGR then cvtColor, matching the video input path.'),
    ('BT-002b', 'classical_ball_v1b_game10', 'ball_tracking/classical-v1b/tracknet-game10-bgr',
     'complete', 'TrackNet game10, 2144 frames, holdout', 'Current default ball baseline; motion prior disabled.'),
    ('BT-002c', 'classical_ball_v1_videos', 'ball_tracking/classical-v1b/demo-0-and-tracknet-clips',
     'complete', 'DemoVideos/0.mp4 and four TrackNet clips', 'Legacy name says v1; videos actually used BGR (v1b). No court gate.'),
    ('BT-003', 'classical_ball_motion_v3_game10', 'ball_tracking/classical-v3-motion/tracknet-game10',
     'superseded', 'TrackNet game10, 2144 frames, holdout', 'Before repair edge-case fixes; outputs identical to BT-004.'),
    ('BT-004', 'classical_ball_motion_v3b_game10', 'ball_tracking/classical-v3b-motion/tracknet-game10',
     'complete', 'TrackNet game10, 2144 frames, holdout', 'Fixed motion prior experiment; optional, not the default.'),
    ('PL-001', 'classical_ball_court_gated_v2', 'pipeline/classical-v2-court-gate/demo-0-and-tracknet-clips',
     'complete', 'DemoVideos/0.mp4, 12113 frames at 25fps; two TrackNet clips', 'Court v3 plus ball v1b; motion prior disabled.'),
    ('PL-002', 'classical_ball_motion_v3_demo', 'pipeline/classical-v3-court-gate-motion/demo-0-interrupted',
     'interrupted', 'DemoVideos/0.mp4', 'Rendering was stopped for repair fixes. MP4 files may be incomplete; do not compare metrics.'),
    ('PL-003', 'classical_ball_motion_v3b_demo', 'pipeline/classical-v3b-court-gate-motion/demo-0',
     'complete', 'DemoVideos/0.mp4, 12113 frames at 25fps', 'Final prior-on demo; reuses PL-001 court decisions. Paired baseline is prior-off.'),
]


def artifact_inventory(directory, hash_paths=None):
    """Check file identity and hash resident files without hydrating iCloud data."""
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.name not in ('manifest.json', 'RUN.md'):
            relative = str(path.relative_to(directory))
            stat = path.stat()
            resident = not bool(getattr(stat, 'st_flags', 0) & 0x40000000)
            digest = None
            should_hash = resident if hash_paths is None else relative in hash_paths
            if should_hash:
                hasher = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        hasher.update(block)
                digest = hasher.hexdigest()
            result[relative] = dict(inode=stat.st_ino, size=stat.st_size,
                                    mtime_ns=stat.st_mtime_ns, sha256=digest)
    return result


def migrate(apply=False):
    """Move known runs and write a navigable output index."""
    # Validate the whole plan before moving anything.
    for _, old, new, *_ in RUNS:
        source, target = OUTPUTS / old, OUTPUTS / new
        if source.is_symlink() and source.resolve() == target.resolve() and target.is_dir():
            continue
        if not source.is_dir() or source.is_symlink() or target.exists():
            raise RuntimeError(f'Unexpected migration state: {source} -> {target}')
    catalog = []
    for run_id, old, new, status, dataset, description in RUNS:
        source, target = OUTPUTS / old, OUTPUTS / new
        print(f'{run_id}: {old} -> {new} [{status}]')
        if not apply:
            continue
        moved = not source.is_symlink()
        inventory = artifact_inventory(source) if moved else None
        if moved:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            try:
                source.symlink_to(new, target_is_directory=True)
            except OSError:
                target.rename(source)
                raise
        if inventory is not None and inventory != artifact_inventory(
                target, {p for p, s in inventory.items() if s['sha256'] is not None}):
            raise RuntimeError(f'Artifact hash mismatch: {run_id}')
        doc = {'court_detection': 'court_detection.md', 'ball_tracking': 'ball_tracking.md',
               'pipeline': 'pipeline.md'}[new.split('/')[0]]
        metadata = {
            'schema_version': 1, 'run_id': run_id, 'target': new.split('/')[0],
            'method_version': new.split('/')[1], 'path': f'data/outputs/{new}',
            'legacy_path': f'data/outputs/{old}', 'status': status, 'dataset': dataset,
            'description': description, 'provenance': 'retrospective; original execution timestamp and full parameters were not recorded',
            'experiment_document': f'docs/experiments/{doc}',
        }
        if inventory is not None:
            metadata['migration_check'] = dict(files=len(inventory),
                sha256_checked=sum(s['sha256'] is not None for s in inventory.values()),
                cloud_only_identity_checked=sum(s['sha256'] is None for s in inventory.values()))
        elif (target / 'manifest.json').exists():
            previous = json.loads((target / 'manifest.json').read_text())
            if 'migration_check' in previous:
                metadata['migration_check'] = previous['migration_check']
        # Preserve the evidence files and keep one authoritative metadata file per run.
        metrics = target / 'metrics.txt'
        if metrics.exists():
            metadata['metrics_file'] = 'metrics.txt'
            metadata['metrics_text'] = metrics.read_text()
        summary = target / '0.summary.json'
        if summary.exists():
            metadata['summary'] = json.loads(summary.read_text())
        (target / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        (target / 'RUN.md').write_text(
            f'# {run_id} — {metadata["method_version"]}\n\n'
            f'- Target: `{metadata["target"]}`\n- Status: **{status}**\n- Data: {dataset}\n'
            f'- {description}\n- [Experiment details]({os.path.relpath(ROOT / metadata["experiment_document"], target)})\n'
            '- Recorded scores: `metrics.txt` / `0.summary.json`, when present.\n'
            '- Full settings not recorded historically are unknown, not assumed from current defaults.\n')
        catalog.append(metadata)
    if apply:
        (OUTPUTS / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
        lines = ['# Experiment outputs', '', 'Start with [experiment documentation](../../docs/experiments/README.md).', '',
                 'Canonical paths are grouped by target / method-version / run. Legacy names are compatibility symlinks.', '',
                 '| Run | Target / version | Status | Data | Results |', '|---|---|---|---|---|']
        for entry in catalog:
            relative = entry['path'].removeprefix('data/outputs/')
            lines.append(f'| {entry["run_id"]} | {entry["target"]} / {entry["method_version"]} | {entry["status"]} | {entry["dataset"]} | [open]({relative}/RUN.md) |')
        (OUTPUTS / 'README.md').write_text('\n'.join(lines) + '\n')
