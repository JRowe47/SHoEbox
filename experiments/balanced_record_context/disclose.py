"""Pre-fit immutable codebook collision and environment disclosure."""
import hashlib,json,platform,subprocess,sys
from pathlib import Path
import numpy as np
from experiments.discriminative_keys.test_objective import code
w=Path(sys.argv[1]);c=np.array([[code(b,r) for r in range(16)] for b in range(256)],dtype='<f4')
collisions=[(b,a) for a in range(256) for b in range(a) if np.array_equal(c[a],c[b])]
r=dict(labels=256,coordinates=16,collision_pairs=collisions,sha256=hashlib.sha256(c.tobytes()).hexdigest(),fixed_readonly_codebook_bytes=c.nbytes,minimum_hamming=min(int(np.count_nonzero(c[a]!=c[b])) for a in range(256) for b in range(a)),training_bytes=0,pairwise_comparisons=32640,policy='All 256 A5 byte codes unchanged; no collisions omitted')
(w/'codebook.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
e=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform(),compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],start_main='3487578a98198e19de833e118cf0d6c5f6b5c0f6',fma='disabled: -ffp-contract=off')
(w/'environment.json').write_text(json.dumps(e,indent=2)+'\n')
