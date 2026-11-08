from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
a=json.loads((root/'MANIFEST.json').read_text())
for n,pin in a['files'].items():
 p=root/n
 assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'],n
print('Verified',len(a['files']),'release files')
