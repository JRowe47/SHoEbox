"""Package exact published source, complete raw evidence and resumable Git history."""
import argparse, hashlib, json, shutil, subprocess, tarfile, tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--commit', required=True)
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
git = lambda *args: subprocess.check_output(['git', *args], cwd=root, text=True).strip()
assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main') == a.commit
assert not git('status', '--porcelain')
stage = a.out / 'SHoEbox_A9_F1_Resume'
stage.mkdir(parents=True, exist_ok=False)
(stage / 'SHoEbox').mkdir()
with subprocess.Popen(['git', 'archive', a.commit], cwd=root, stdout=subprocess.PIPE) as proc:
    with tarfile.open(fileobj=proc.stdout, mode='r|') as tf:
        tf.extractall(stage / 'SHoEbox', filter='data')
    assert proc.wait() == 0
subprocess.run(['git', 'bundle', 'create', str((stage / 'repository.bundle').resolve()), '--all'], cwd=root, check=True)
subprocess.run(['git', 'bundle', 'verify', str((stage / 'repository.bundle').resolve())], cwd=root, check=True)
ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.o', '.git')
shutil.copytree(a.workspace, stage / 'a9_f1_work', ignore=ignore)
shutil.copytree(a.baseline, stage / 'baseline', ignore=ignore)
(stage / 'PUBLISHED_COMMIT.txt').write_text(a.commit + '\n')
(stage / 'PUBLICATION.json').write_text(json.dumps(dict(
    published_commit=a.commit,
    verified_remote_tree=git('rev-parse', 'HEAD^{tree}'),
    url='https://github.com/JRowe47/SHoEbox/commit/' + a.commit,
    prefit_local='00d5cb393c075710925a6fd9add8606915ca3a5a',
    prefit_published='7dc8b5c275d00b469f068d788064b63025833120',
    original_main='fa54cbd1c8c90be756ea4da8cfef2ccb735ed183',
    research_fitting_bytes=20250000,
    audit_opened=False), indent=2) + '\n')
(stage / 'RESTORE.md').write_text('''# Restore A9-F1

Extract the archive, enter `SHoEbox_A9_F1_Resume/`, and verify its payload:

```bash
sha256sum -c SHA256SUMS
git -C SHoEbox init -b recovery
git -C SHoEbox fetch ../repository.bundle 'refs/heads/*:refs/heads/*'
git -C SHoEbox symbolic-ref HEAD refs/heads/main
git -C SHoEbox reset --mixed main
git -C SHoEbox status --short
```

The exported source already matches main; reset changes the index only.
The local pre-fit commit is also retained as a branch. Use the SHA in
`PUBLISHED_COMMIT.txt` to verify main. Python with NumPy/Matplotlib and a C++17
compiler is sufficient for the archived tooling; native libraries rebuild with
FMA disabled. The completed 20.25M fitting budget must not be repeated.

Read `SHoEbox/reports/exp0305a9f1/RESUME.md`. The delivery verifier reads files
and hashes only; it does not create a model, run inference or perform fitting:

```bash
cd SHoEbox
python reports/exp0305a9f1/verify_delivery.py ../a9_f1_work
```

The outside delivery manifest contains the archive digest and packaging/check
CPU receipts, which cannot be recursively embedded in the archive they hash.
All research, correctness and diagnostic receipts are inside the archive.
''')
with tempfile.TemporaryDirectory(prefix='a9_restore_check_', dir=a.out) as tmp:
    restored = Path(tmp) / 'SHoEbox'
    subprocess.run(['git', 'clone', '--quiet', '-b', 'main', str((stage / 'repository.bundle').resolve()), str(restored)], check=True)
    subprocess.run(['python', str((restored / 'reports/exp0305a9f1/verify_delivery.py').resolve()), str((stage / 'a9_f1_work').resolve())], cwd=restored, check=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
files = sorted(p for p in stage.rglob('*') if p.is_file())
(stage / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(stage)}\n' for p in files))
archive = a.out / 'SHoEbox_A9_F1_Resumable.tar.xz'
assert not archive.exists()
with tarfile.open(archive, 'w:xz', preset=6) as tf:
    tf.add(stage, arcname=stage.name)
expected = {str(p.relative_to(stage)): sha(p) for p in files}
seen = {}
with tarfile.open(archive, 'r:xz') as tf:
    for member in tf:
        if not member.isfile():
            continue
        name = str(Path(member.name).relative_to(stage.name))
        if name != 'SHA256SUMS':
            seen[name] = hashlib.sha256(tf.extractfile(member).read()).hexdigest()
assert seen == expected
receipt = dict(archive=archive.name, sha256=sha(archive), bytes=archive.stat().st_size,
               payload_files=len(expected), every_member_hash_verified=True,
               restored_git_bundle_delivery_check_passed=True,
               published_commit=a.commit, research_fitting_bytes=20250000)
(a.out / 'archive_verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
