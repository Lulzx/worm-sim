#!/usr/bin/env python3
"""Inspect one preselected candidate NWB object's names/shapes, never dataset values."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import h5py


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class RangeFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url=url;self.size=size;self.position=0;self.ranges=[];self.total=0;self.cache={}
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if position<0:raise ValueError('invalid seek')
        self.position=position;return position
    def readinto(self,b):
        raw=self.read(len(b));b[:len(raw)]=raw;return len(raw)
    def read(self,length=-1):
        if length<0:raise ValueError('unbounded reads are disabled')
        length=min(length,max(0,self.size-self.position))
        if not length:return b''
        start=self.position;end=start+length-1
        if (start,length) in self.cache:
            self.position+=length;return self.cache[(start,length)]
        if length>65536 or self.total+length>2_000_000 or len(self.ranges)>=500:
            raise ValueError(f'schema read budget exceeded: requested={length}, received={self.total}, requests={len(self.ranges)}')
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=25) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                raise ValueError('server did not honor exact bounded range')
            raw=response.read(length+1)
        if len(raw)!=length:raise ValueError('range length differs')
        self.ranges.append({'start':start,'bytes':length,'sha256':digest(raw)})
        self.cache[(start,length)]=raw
        self.total+=length;self.position+=length;return raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    p.add_argument('--group',action='append',help='Explicit root group; defaults to stimulus and intervals')
    a=p.parse_args()
    if Path(a.output).exists():raise ValueError('output already exists')
    selected=a.group or ['stimulus','intervals']
    if len(set(selected))!=len(selected) or any('/' in name or not name for name in selected):raise ValueError('choose distinct root group names')
    raw=Path(a.inventory).read_bytes();inventory=json.loads(raw)
    if inventory['dandiset']!='001623' or inventory['version']!='0.251015.0312':raise ValueError('unexpected inventory')
    # Pick using names only, before opening any object. Do not choose by outcomes.
    record=min(inventory['records'],key=lambda r:r['path']);asset=record['asset_id']
    with urllib.request.urlopen('https://api.dandiarchive.org/api/assets/'+asset+'/',timeout=25) as response:metadata=response.read()
    meta=json.loads(metadata)
    if meta['path']!=record['path'] or meta['contentSize']!=record['bytes']:raise ValueError('asset differs from inventory')
    url=next(u for u in meta['contentUrl'] if u.startswith('https://dandiarchive.s3.amazonaws.com/blobs/'))
    rows=[];roots=[];missing=[];error=None
    with RangeFile(url,record['bytes']) as remote:
        try:
            with h5py.File(remote,'r') as file:
                roots=list(file.keys())
                def inspect(name,obj):
                    row={'path':name,'kind':'dataset' if isinstance(obj,h5py.Dataset) else 'group'}
                    if isinstance(obj,h5py.Dataset):row.update(shape=obj.shape,dtype=str(obj.dtype))
                    rows.append(row)
                    if len(rows)>5000:raise ValueError('schema object budget exceeded')
                for name in selected:
                    if name not in roots:
                        missing.append(name);continue
                    group=file[name];inspect(name,group)
                    if isinstance(group,h5py.Group):
                        group.visititems(lambda sub,obj:inspect(name+'/'+sub,obj))
        except Exception as exc:
            error={'type':type(exc).__name__,'message':str(exc)}
        receipt={'schema_version':1,'inventory_sha256':digest(raw),'selection':'lexicographically first asset path',
            'status':'incomplete' if error else 'selected_groups_inspected','error':error,
            'selected_groups':selected,'root_names':roots,'missing_selected_groups':missing,
            'recording':record,'asset_metadata_sha256':digest(metadata),'published_asset_digest':meta['digest'],
            'published_asset_digest_independently_verified':False,'url':url,'h5py':h5py.__version__,
            'hdf5':h5py.version.hdf5_version,'script_sha256':digest(Path(__file__).read_bytes()),
            'bytes_received':remote.total,'ranges':remote.ranges,'objects':rows,
            'scope':'One candidate asset, explicit root groups: HDF5 group/dataset names, shapes and dtype only; no dataset indexing or attribute values read. Range responses may contain bytes colocated with metadata; no neural response values decoded or displayed. No cohort-wide eligibility or independence claim.'}
    with Path(a.output).open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ['status','error','root_names','missing_selected_groups','objects','bytes_received']},indent=2))
    if error:raise SystemExit(1)


if __name__=='__main__':main()
