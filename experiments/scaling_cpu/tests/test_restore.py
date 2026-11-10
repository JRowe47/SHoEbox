from pathlib import Path
import gzip,hashlib,json,tempfile,unittest
from unittest.mock import patch
import restore_data


class RestoreTests(unittest.TestCase):
    def fixture(self,root):
        h=lambda raw:hashlib.sha256(raw).hexdigest();splits={};payloads={}
        for index,name in enumerate(('calibration','train','validation','test')):
            raw=(name+' document').encode();payload=raw+b'\n\n';identity=str(index);payloads[identity]=payload
            splits[name]=dict(targets=len(payload)-1,sha256=h(payload),documents=[dict(id=identity,source_sha256=h(raw),
                source_file='sample/10BT/000_00000.parquet',row_idx=index,start=0,source_byte_offset=0,bytes_used=len(payload),segment_sha256=h(payload))])
        raw=json.dumps(dict(revision='a'*40,splits=splits),indent=2).encode();manifest=root/'source.json.gz';manifest.write_bytes(gzip.compress(raw))
        return manifest,raw,payloads

    def test_restore_exact_bytes_and_resume_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,raw,payloads=self.fixture(root);out=root/'data'
            with patch.object(restore_data,'fetch_file',return_value=payloads) as fetch:
                result=restore_data.restore(manifest,out);self.assertEqual(len(result['restored']),4);self.assertEqual(fetch.call_count,1)
                self.assertEqual((out/'manifest.json').read_bytes(),raw)
                second=restore_data.restore(manifest,out);self.assertEqual(second['restored'],[]);self.assertEqual(fetch.call_count,1)
            (out/'train.bin').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'Existing split differs'):restore_data.restore(manifest,out)

    def test_wrong_download_never_commits_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,raw,payloads=self.fixture(root);payloads['0']=b'wrong';out=root/'data'
            with patch.object(restore_data,'fetch_file',return_value=payloads),self.assertRaisesRegex(ValueError,'Segment hash'):
                restore_data.restore(manifest,out)
            self.assertFalse((out/'manifest.json').exists())


if __name__=='__main__':unittest.main()
