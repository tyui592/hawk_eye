"""Create a reviewed court dataset copy using accepted v3 label corrections."""

import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    """Hash a file."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(args):
    if args.destination.exists():
        raise ValueError(f'destination already exists: {args.destination}')
    source_resolved = args.source.resolve()
    destination_resolved = args.destination.resolve()
    if source_resolved == destination_resolved or source_resolved in destination_resolved.parents:
        raise ValueError('destination must be outside the source dataset')
    source_val = args.source / 'data_val.json'
    source_train = args.source / 'data_train.json'
    before = {'data_train.json': sha256(source_train), 'data_val.json': sha256(source_val)}

    results = json.loads(args.results.read_text())
    corrections = [r for r in results if r['num_wrong'] > 0]
    if len(corrections) != 102 or len({r['id'] for r in corrections}) != 102:
        raise RuntimeError(f'expected 102 unique reviewed failures, got {len(corrections)}')
    if any(len(r['keypoints']) != 14 for r in corrections):
        raise RuntimeError('every correction must contain 14 court keypoints')

    # APFS copy-on-write gives an independent full dataset path without duplicating 7.2 GB.
    # If either copy is later modified, only the changed blocks are separated.
    subprocess.run(['cp', '-cR', str(args.source), str(args.destination)], check=True)
    source_image_count = sum(1 for _ in (args.source / 'images').glob('*.png'))
    copied_image_count = sum(1 for _ in (args.destination / 'images').glob('*.png'))
    if copied_image_count != source_image_count:
        raise RuntimeError(f'image copy is incomplete: {copied_image_count} != {source_image_count}')
    copied_val = args.destination / 'data_val.json'
    labels = json.loads(copied_val.read_text())
    by_id = {row['id']: row for row in labels}
    missing = sorted(r['id'] for r in corrections if r['id'] not in by_id)
    if missing:
        raise RuntimeError(f'correction ids missing from val labels: {missing[:3]}')

    audit = []
    for result in corrections:
        row = by_id[result['id']]
        old = row['kps']
        new = result['keypoints']
        row['kps'] = new
        audit.append({
            'id': result['id'], 'old_kps': old, 'new_kps': new,
            'v3_num_wrong_against_original': result['num_wrong'],
            'v3_court_score': result['court_score'],
        })
    copied_val.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + '\n')

    after_source = {'data_train.json': sha256(source_train), 'data_val.json': sha256(source_val)}
    if before != after_source:
        raise RuntimeError('source labels changed while creating the reviewed copy')
    if sha256(args.destination / 'data_train.json') != before['data_train.json']:
        raise RuntimeError('train labels must remain byte-identical')

    provenance = {
        'schema_version': 1,
        'dataset': args.destination.name,
        'source_dataset': str(args.source),
        'source_label_sha256': before,
        'correction_source': str(args.results),
        'correction_source_sha256': sha256(args.results),
        'review_decision': ('The project owner visually reviewed all 102 v3 val failures and '
                            'accepted the v3 court projection as more accurate than the original GT.'),
        'correction_policy': ('For each reviewed val image with num_wrong > 0, replace all 14 kps '
                              'with the full-precision v3 keypoints to preserve one coherent homography.'),
        'corrected_split': 'val',
        'corrected_images': len(audit),
        'copied_images': copied_image_count,
        'train_labels': 'unchanged and byte-identical to source',
        'images': 'independent APFS copy-on-write clones of the source image files',
        'corrections': audit,
    }
    (args.destination / 'label_corrections_v3.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    (args.destination / 'README.md').write_text(
        '# TennisCourtDetector v3-reviewed labels\n\n'
        'This is a separate APFS copy-on-write clone of `data/TennisCourtDetector`.\n'
        'The source dataset is unchanged. Train labels are byte-identical to the source.\n'
        'In val, all 14 keypoints of the 102 owner-reviewed v3 failure images were replaced\n'
        'with the full-precision v3 predictions. See `label_corrections_v3.json` for every\n'
        'old/new label and the source result hash. These are reviewed pseudo-labels, so a v3\n'
        'score on the corrected val split measures self-consistency, not independent accuracy.\n')

    print(f'Created {args.destination}: {len(labels)} val rows, {len(audit)} corrected images')
    print(f'Original labels unchanged: {before == after_source}')
