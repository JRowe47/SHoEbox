"""Explicit E8 research-to-public conversion with preserved external cursor."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from .model import E8Model


def convert(source, destination, *, cursor=None):
    source, destination = Path(source), Path(destination)
    cursor = Path(cursor) if cursor else source.with_suffix('.cursor.json')
    raw, cursor_raw = source.read_bytes(), cursor.read_bytes()
    state = json.loads(cursor_raw)
    decoder = None
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if set(z.namelist()) != {'adapter.qr', 'decoder.npz'}:
                raise ValueError('unexpected research reconstruction container')
            payload, decoder = z.read('adapter.qr'), z.read('decoder.npz')
    else:
        payload = raw
    with E8Model.import_research(payload)[0] as model:
        converted = model.dumps()
        if state['model_bytes'] != model.stats()['bytes']:
            raise ValueError('checkpoint/cursor mismatch')
    if decoder is not None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('model.shoe', converted)
            z.writestr('decoder.npz', decoder)
        output = buf.getvalue()
    else:
        output = converted
    sha = lambda x: hashlib.sha256(x).hexdigest()
    receipt = dict(source=str(source), destination=str(destination),
                   source_format='EXP01-E8-research-1', destination_format='e8-public-1',
                   source_sha256=sha(raw), destination_sha256=sha(output),
                   source_payload_sha256=sha(payload), destination_payload_sha256=sha(converted),
                   cursor_sha256=sha(cursor_raw), payload_after_magic_exact=converted[48:]==payload[40:],
                   decoder_sha256=sha(decoder) if decoder is not None else None)
    paths = [destination, destination.with_suffix('.cursor.json'), destination.with_suffix('.import.json')]
    if len(set(paths)) != 3 or any(p.exists() for p in paths):
        raise FileExistsError('choose unused destination/checkpoint, cursor and receipt paths')
    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = [output, cursor_raw, (json.dumps(receipt, indent=2)+'\n').encode()]
    for path, data in zip(paths, contents):
        with path.open('xb') as f:
            f.write(data)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--cursor',type=Path)
    a=p.parse_args();print(json.dumps(convert(a.source,a.destination,cursor=a.cursor),indent=2))


if __name__ == '__main__':
    main()
