#!/usr/bin/env python3
"""Read the pinned Dunn ZIP directory without fetching/decompressing its members."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    metadata_url='https://zenodo.org/api/records/17353307'
    with urllib.request.urlopen(metadata_url,timeout=25) as response:metadata=response.read()
    info=json.loads(metadata)
    file=next(f for f in info['files'] if f['key']=='intermediate_datafiles.zip')
    if file['id']!='80c6cae6-cad6-4a7b-8271-91e0e541860f':
        raise ValueError('archive identity changed')
    size=file['size'];url=file['links']['self']
    def read_range(start,length):
        request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+length-1}'})
        with urllib.request.urlopen(request,timeout=25) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{start+length-1}/{size}':
                raise ValueError('server did not honor exact metadata byte range')
            raw=response.read(length+1)
        if len(raw)!=length:raise ValueError('metadata range length mismatch')
        return raw
    # This particular pinned archive has no ZIP comment; reject other layouts.
    end=read_range(size-22,22)
    magic,disk,cd_disk,disk_count,count,cd_size,offset,comment=struct.unpack('<4s4H2IH',end)
    if magic!=b'PK\x05\x06' or disk or cd_disk or disk_count!=count or comment or cd_size>2000000 or offset+cd_size!=size-22:
        raise ValueError('unsupported ZIP directory layout')
    directory=read_range(offset,cd_size);cursor=0;files=[]
    while cursor<len(directory):
        row=struct.unpack_from('<4s6H3I5H2I',directory,cursor)
        if row[0]!=b'PK\x01\x02':raise ValueError('invalid directory entry')
        n,x,c=row[10:13]
        name=directory[cursor+46:cursor+46+n].decode('utf-8' if row[3]&0x800 else 'cp437')
        files.append({'name':name,'compressed_bytes':row[8],'uncompressed_bytes':row[9],'crc32':f'{row[7]:08x}'})
        cursor+=46+n+x+c
    if cursor!=len(directory) or len(files)!=count:raise ValueError('directory count mismatch')
    receipt={'schema_version':1,'metadata_url':metadata_url,'metadata_sha256':hashlib.sha256(metadata).hexdigest(),
        'doi':info['doi'],'archive_url':url,'archive_bytes':size,'published_archive_checksum':file['checksum'],
        'archive_checksum_independently_verified':False,'central_directory_sha256':hashlib.sha256(directory).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bytes_requested_from_archive':22+cd_size,
        'file_entries':files,'scope':'Exact ZIP end record and central directory ranges only; no member payloads requested or decompressed. Filenames establish inventory, not contents or cohort independence.'}
    with Path(a.output).open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print('Directory entries:',count,'archive metadata bytes:',22+cd_size)

if __name__=='__main__':main()
