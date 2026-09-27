import io
import unittest
from unittest.mock import patch
from inspect_dunn_nwb_schema import RangeFile


class RangeTests(unittest.TestCase):
    def test_seek_and_exact_reads(self):
        payload=b'abcdefghij'
        def serve(request,timeout):
            lo,hi=map(int,request.get_header('Range').removeprefix('bytes=').split('-'))
            result=io.BytesIO(payload[lo:hi+1]);result.status=206
            result.headers={'Content-Range':f'bytes {lo}-{hi}/{len(payload)}'}
            return result
        with patch('urllib.request.urlopen',side_effect=serve):
            with RangeFile('https://example.invalid',len(payload)) as f:
                self.assertEqual(f.read(3),b'abc');f.seek(0);self.assertEqual(f.read(3),b'abc');f.seek(-2,2)
                b=bytearray(4);self.assertEqual(f.readinto(b),2);self.assertEqual(b[:2],b'ij')
                self.assertEqual(f.read(1),b'')
                self.assertEqual(f.total,5)
                self.assertEqual([r['start'] for r in f.ranges],[0,8])

    def test_reject_full_body_before_read_and_enforce_budget(self):
        class RefuseRead(io.BytesIO):
            status=200;headers={}
            def read(self,*args):raise AssertionError('body must not be read')
        with patch('urllib.request.urlopen',return_value=RefuseRead()):
            with RangeFile('https://example.invalid',100) as f:
                with self.assertRaises(ValueError):f.read(10)
        with RangeFile('https://example.invalid',3_000_000) as f:
            for amount in [-1,65537]:
                with self.assertRaises(ValueError):f.read(amount)
            f.total=2_000_000
            with self.assertRaises(ValueError):f.read(1)




class SchemaScopeTests(unittest.TestCase):
    def test_only_selected_structure_is_read(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        import h5py
        from inspect_dunn_nwb_schema import main
        buffer=io.BytesIO()
        with h5py.File(buffer,'w') as f:
            f.create_dataset('acquisition/response',data=[123456.,654321.])
            f.create_dataset('stimulus/presentation/pulse/data',data=[0.,1.])
        raw=buffer.getvalue()
        class FakeFile(io.BytesIO):
            def __init__(self,*args):super().__init__(raw);self.total=len(raw);self.ranges=[]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inventory=root/'inventory.json';output=root/'result.json'
            record={'path':'first.nwb','asset_id':'test','bytes':len(raw)}
            inventory.write_text(json.dumps({'dandiset':'001623','version':'0.251015.0312','records':[record]}))
            meta={'path':'first.nwb','contentSize':len(raw),'contentUrl':['https://dandiarchive.s3.amazonaws.com/blobs/test'],'digest':{}}
            with patch.object(sys,'argv',['inspect','--inventory',str(inventory),'--output',str(output)]), patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(meta).encode())), patch('inspect_dunn_nwb_schema.RangeFile',FakeFile), patch.object(h5py.Dataset,'__getitem__',side_effect=AssertionError('dataset values must not be read')), patch('builtins.print'):
                main()
            result=json.loads(output.read_text())
            self.assertEqual(result['status'],'selected_groups_inspected')
            self.assertEqual(result['missing_selected_groups'],['intervals'])
            self.assertTrue(all(r['path'].startswith('stimulus') for r in result['objects']))
            self.assertNotIn('123456',output.read_text())
            self.assertEqual(result['objects'][-1]['shape'],[2])

    def test_failed_hdf_open_retains_incomplete_receipt(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        from inspect_dunn_nwb_schema import main
        class FakeFile(io.BytesIO):
            def __init__(self,*args):super().__init__();self.total=8;self.ranges=[{'start':0,'bytes':8}]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inventory=root/'inventory.json';output=root/'result.json'
            record={'path':'first.nwb','asset_id':'test','bytes':8}
            inventory.write_text(json.dumps({'dandiset':'001623','version':'0.251015.0312','records':[record]}))
            meta={'path':'first.nwb','contentSize':8,'contentUrl':['https://dandiarchive.s3.amazonaws.com/blobs/test'],'digest':{}}
            with patch.object(sys,'argv',['inspect','--inventory',str(inventory),'--output',str(output)]), patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(meta).encode())), patch('inspect_dunn_nwb_schema.RangeFile',FakeFile), patch('inspect_dunn_nwb_schema.h5py.File',side_effect=OSError('synthetic failure')), patch('builtins.print'):
                with self.assertRaises(SystemExit) as stopped:main()
                self.assertEqual(stopped.exception.code,1)
            result=json.loads(output.read_text())
            self.assertEqual(result['status'],'incomplete')
            self.assertEqual(result['bytes_received'],8)
            self.assertEqual(result['error']['type'],'OSError')

if __name__=='__main__':unittest.main()
