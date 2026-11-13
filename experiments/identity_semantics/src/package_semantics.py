"""Build a verified source, evidence, data, and checkpoint archive."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
PREFIX='SHoEbox_Identity_Semantics/'
ARCHIVE='SHoEbox_Identity_Semantics_Source_and_Checkpoints.zip'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def include(p):
 r=p.relative_to(ROOT)
 if '__pycache__' in r.parts or any(v.startswith('.') for v in r.parts):return False
 if r.parts[0] in ['runtime','src','tests']:
  return p.suffix in ['.py','.cpp','.inc','.json'] or (r.parts[:2]==('runtime','configs') and p.suffix=='.npz')
 if r.parts[0]=='results':return p.suffix in ['.json','.log'] or (p.name=='last.npz' and r.parts[1].startswith('full_'))
 if r.parts[0]=='data':return p.name in ['train_segment.txt','heldout.bin','source_manifest.json']
 if r.parts[0]=='provenance':return p.suffix in ['.md','.json','.patch']
 if r.parts[0]=='deliverables':return p.suffix in ['.md','.pdf','.json','.csv','.png']
 return len(r.parts)==1 and p.name in ['README.md','requirements.txt','requirements-reports.txt','BINDING_SCHEMA.json','DATA_MANIFEST.json','SOURCE_MANIFEST.json','ARTIFACT_MANIFEST.json']
source={str(p.relative_to(ROOT)):digest(p) for folder in ['runtime','src','tests'] for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and include(p)}
write(ROOT/'SOURCE_MANIFEST.json',dict(description='Tested consistent identity/vector semantics experiment; source/configuration hashes.',base_repository='https://github.com/JRowe47/SHoEbox',base_commit='6d06d60072c3032ac1ce13ceed04ee8f7d89a4cc',base_path='experiments/composition_repair',semantic_checkpoint_version=30,semantic_composition_state_abi=4,comparator_checkpoint_version=29,comparator_composition_state_abi=3,parameters=1031936,files=source))
artifacts={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and include(p) and (p.relative_to(ROOT).parts[0]=='data' or (p.name=='last.npz' and p.relative_to(ROOT).parts[0]=='results'))}
write(ROOT/'ARTIFACT_MANIFEST.json',dict(archive=ARCHIVE,archive_prefix=PREFIX,description='Per-file hashes avoid a self-referential archive checksum. SHA256SUMS also covers source, reports, and raw measurements.',files=artifacts))
files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and include(p)]
(ROOT/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(ROOT)}\n' for p in files));files.append(ROOT/'SHA256SUMS')
archive=ROOT/'deliverables'/ARCHIVE
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,PREFIX+str(p.relative_to(ROOT)))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for p in files:assert hashlib.sha256(z.read(PREFIX+str(p.relative_to(ROOT)))).hexdigest()==digest(p)
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files),sha256=digest(archive)),indent=2))
