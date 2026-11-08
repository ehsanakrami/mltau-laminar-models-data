from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[1]
a=json.loads((root/'MANIFEST.json').read_text());expected=set(a['files'])|{'MANIFEST.json'}
if (root/'.git').exists():
 actual=set(subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().rstrip('\0').split('\0'))
else:
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and not any(x in {'__pycache__','.venv','build','dist'} or x.endswith('.egg-info') for x in p.relative_to(root).parts)}
assert actual==expected,{'unlisted':sorted(actual-expected),'missing':sorted(expected-actual)}
for n,pin in a['files'].items():
 p=root/n
 assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'],n
print('Verified exact inventory and SHA256 of',len(a['files']),'release files')
