"""Export the verified publication and all resumable A8 evidence, without binaries."""
import hashlib, json, shutil, subprocess, sys, tarfile
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
workspace = repo.parent
out = workspace / 'deliverables'
stage = workspace / 'delivery_stage'
assert not stage.exists(), 'Use a new output directory; never replace an evidence archive.'
out.mkdir(exist_ok=True)
stage.mkdir()
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
publication = json.loads((workspace / 'publication.json').read_text())
assert publication['commit'] == commit and publication['remote_verified']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo)

def export(revision, destination):
    destination.mkdir()
    proc = subprocess.Popen(['git', 'archive', revision], cwd=repo, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
        archive.extractall(destination, filter='data')
    assert proc.wait() == 0

export(commit, stage / 'SHoEbox')
export('3487578a98198e19de833e118cf0d6c5f6b5c0f6', stage / 'baseline')
subprocess.run(['git', 'bundle', 'create', str(stage / 'source_history.bundle'), '--all'],
               cwd=repo, check=True)
subprocess.run(['git', 'bundle', 'verify', str(stage / 'source_history.bundle')],
               cwd=repo, check=True)
shutil.copytree(workspace / 'a8_work', stage / 'a8_work',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.o'))
shutil.copy2(workspace / 'publication.json', stage / 'publication.json')
(stage / 'PUBLISHED_COMMIT.txt').write_text(commit + '\n')
assert len(list((stage / 'a8_work/a8').glob('*/*.qr'))) == 18
assert not list(stage.rglob('*.so')) and not list(stage.rglob('*.o'))
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
members = {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}
(stage / 'SHA256SUMS.json').write_text(json.dumps(members, indent=2) + '\n')
archive_path = out / 'SHoEbox_A8_Code_and_Evidence.tar.xz'
assert not archive_path.exists()
with tarfile.open(archive_path, 'w:xz', preset=6) as archive:
    for p in sorted(stage.rglob('*')):
        if p.is_file():
            archive.add(p, arcname=str(p.relative_to(stage)), recursive=False)
# Verify every actual compressed member, including complete checkpoint bytes.
with tarfile.open(archive_path, 'r:xz') as archive:
    actual = {}
    for member in archive:
        data = archive.extractfile(member).read()
        actual[member.name] = hashlib.sha256(data).hexdigest()
assert actual == {**members, 'SHA256SUMS.json': digest(stage / 'SHA256SUMS.json')}

report = repo / 'reports/exp0305a8'
for name in ('SHoEbox_EXP0305_A8_Report.md', 'DIAGNOSIS.md', 'A9_REGISTRATION.md',
             'NEXT_CHAT_PROMPT.md', 'RESUME.md'):
    destination = {'DIAGNOSIS.md': 'SHoEbox_A8_Diagnosis.md',
                   'A9_REGISTRATION.md': 'SHoEbox_A9_Registration.md',
                   'NEXT_CHAT_PROMPT.md': 'SHoEbox_A8_Next_Chat_Prompt.md',
                   'RESUME.md': 'SHoEbox_A8_Resume.md'}.get(name, name)
    shutil.copy2(report / name, out / destination)
for p in report.glob('SHoEbox_EXP0305_A8_*.png'):
    shutil.copy2(p, out / p.name)
# Add the verified final identity to ordinary downloads without circular
# self-references inside the committed repository or archive manifest.
with (out / 'SHoEbox_EXP0305_A8_Report.md').open('a') as f:
    f.write('\nVerified publication: [' + commit + '](' + publication['url'] + ').\n')
prompt = out / 'SHoEbox_A8_Next_Chat_Prompt.md'
prompt.write_text(prompt.read_text().replace(
    'Start from latest main, the commit containing this completed A8 report.',
    'Start from latest main, currently\n' + commit + '.\nVerified commit:\n' + publication['url']))
manifest = dict(commit=commit, url=publication['url'], archive_members=len(actual),
                checkpoint_cursor_pairs=18, files={p.name: dict(bytes=p.stat().st_size,
                sha256=digest(p)) for p in sorted(out.iterdir()) if p.is_file()})
(out / 'SHoEbox_A8_Delivery_Manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
