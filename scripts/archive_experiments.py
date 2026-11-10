"""Export compact experiment evidence into the repository for git submission."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 2 * 1024 * 1024


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()
                if not (k == 'first_logits' and isinstance(v, list))}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def archive(source, archive_root=ROOT / 'experiment_results'):
    source = Path(source).resolve()
    destination = Path(archive_root).resolve() / source.name
    if destination == source or source in destination.parents:
        raise ValueError('Archive must be outside the raw run directory')
    destination.mkdir(parents=True, exist_ok=False)
    exported, omitted, shortened = [], [], []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.suffix not in ('.md', '.csv', '.json', '.log'):
            omitted.append(relative.as_posix())
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            value = json.loads(path.read_text(encoding='utf-8'))
            text = json.dumps(compact(value), ensure_ascii=False, indent=2)
            if len(text.encode('utf-8')) > LIMIT:
                omitted.append(relative.as_posix())
                continue
            target.write_text(text + '\n', encoding='utf-8')
        elif path.suffix == '.log':
            with path.open('rb') as stream:
                size = path.stat().st_size
                if size > LIMIT:
                    stream.seek(size - LIMIT)
                    stream.readline()
                    shortened.append(relative.as_posix())
                target.write_bytes(stream.read())
        elif path.stat().st_size <= LIMIT:
            shutil.copyfile(path, target)
        else:
            omitted.append(relative.as_posix())
            continue
        exported.append(relative.as_posix())
    (destination / 'archive_manifest.json').write_text(json.dumps({
        'exported': exported, 'omitted': omitted, 'tail_only_logs': shortened,
        'note': 'Full first_logits arrays and source.zip remain in raw server results. '
                'Numerical comparisons are retained in report.json. This archive is for analysis, not a replacement for raw evidence.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--archive-root', type=Path, default=ROOT / 'experiment_results')
    args = parser.parse_args()
    print(archive(args.directory, args.archive_root))


if __name__ == '__main__':
    main()
