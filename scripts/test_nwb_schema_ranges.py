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
                self.assertEqual(f.read(3),b'abc');f.seek(-2,2)
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


if __name__=='__main__':unittest.main()
