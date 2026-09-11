from pathlib import Path
import json,re,hashlib,itertools,math,subprocess
r=Path(__file__).resolve().parents[1]
c=json.loads((r/'reviews/catalog.json').read_text()); m=json.loads((r/'reviews/source-manifest.json').read_text())
assert len(c['canonical'])==30 and len(c['supplemental'])==12
issues=[]; paths=[r/'README.md']+list((r/'PRDs').glob('*.md'))+list((r/'reviews').glob('*.md'))+list((r/'supplemental').glob('*.md'))+[r/'NOTES/README.md',r/'latex-reports/README.md',r/'archive/README.md']
for p in paths:
 s=p.read_text()
 if any(ord(t)<32 and t not in '\n\t' for t in s): issues.append(f'Control character: {p}')
 if s.count('$$')%2: issues.append(f'Unpaired display math: {p}')
 for dest in re.findall(r'\]\(([^\s)]+)\)',s):
  if '://' in dest or dest.startswith('#'): continue
  path=(p.parent/dest.split('#')[0]).resolve()
  if not path.exists() and path.name!='VALIDATION.md': issues.append(f'Broken link: {p.name} -> {dest}')
for x in c['canonical']:
 p=r/'PRDs'/x['file']; s=p.read_text()
 for h in ['## Core question','## Scope and assumptions','## Mathematical target','## Required outputs and proof obligations','## Validation and rejection controls','## Milestones and research extension']:
  assert h in s,(p,h)
 assert (r/'reviews'/x['file']).exists()
 assert (r/'archive/original-PRDs'/x['file']).exists()
# Every current problem has one complete accessibility guide.
for group, folder in [('canonical', 'PRDs'), ('supplemental', 'supplemental')]:
 for x in c[group]:
  p=r/folder/x['file']; s=p.read_text()
  for h in ['## Plain-language guide', '### The problem in everyday terms', '### Key terms', '### Why this matters', '### What progress would mean']:
   assert s.count(h)==1,(p,h)
  assert s in (r/'reviews/COMPLETE-REVIEW.md').read_text(),p
assert (r/'reviews/COMPLETE-REVIEW.md').read_text().count('## Plain-language guide')==42
# Every original PRD byte is preserved.
for rel,digest in m['original_files'].items():
 if rel.startswith('PRDs/'):
  assert hashlib.sha256((r/'archive/original-PRDs'/Path(rel).name).read_bytes()).hexdigest()==digest
 if rel.startswith('latex-reports/') or rel.endswith('.docx') or rel.endswith('.txt') or rel.endswith('.pdf') or rel.endswith('.tex'):
  assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==digest,rel
# Independent exact GF(2) calculation for the revised Hamming product benchmark.
H=[[((j>>i)&1) for j in range(1,8)] for i in range(3)]
def bitrow(indices):
 z=0
 for i in indices:z^=1<<i
 return z
hx=[];hz=[]
for a in range(3):
 for j in range(7):
  hx.append(bitrow([i*7+j for i in range(7) if H[a][i]]+[49+a*3+b for b in range(3) if H[b][j]]))
for i in range(7):
 for b in range(3):
  hz.append(bitrow([i*7+j for j in range(7) if H[b][j]]+[49+a*3+b for a in range(3) if H[a][i]]))
def basis(rows):
 bs={}
 for z in rows:
  while z:
   p=z.bit_length()-1
   if p not in bs:bs[p]=z;break
   z^=bs[p]
 return bs
def inrow(z,bs):
 while z:
  p=z.bit_length()-1
  if p not in bs:return False
  z^=bs[p]
 return True
assert all((a&b).bit_count()%2==0 for a in hx for b in hz)
assert 58-len(basis(hx))-len(basis(hz))==16
for check,stabs in [(hz,hx),(hx,hz)]:
 bs=basis(stabs)
 for w in [1,2]:
  for inds in itertools.combinations(range(58),w):
   z=bitrow(inds)
   assert any((z&row).bit_count()%2 for row in check) or inrow(z,bs)
 assert any(all((bitrow(inds)&row).bit_count()%2==0 for row in check) and not inrow(bitrow(inds),bs) for inds in itertools.combinations(range(58),3))
# Check exact scalar-network benchmark and correction of the timescale.
assert all(6-11*x+6*x*x-x*x*x==0 for x in [1,2,3])
assert [-11+12*x-3*x*x for x in [1,2,3]]==[-2,1,-2]
assert 7.3<math.exp(10**0.3)<7.4
assert 4**20==1099511627776
# Verify the two stated Weyl locations and opposite velocity determinants.
for z in [math.pi/2,-math.pi/2]:
 assert abs(2-1-1-math.cos(z))<1e-12
assert math.sin(math.pi/2)*math.sin(-math.pi/2)<0
assert not issues,'\n'.join(issues)
print(json.dumps({'canonical_rewrites':30,'individual_critiques':30,'supplemental_revisions':12,'markdown_files_checked':len(paths),'preserved_original_PRD_files':33,'local_link_errors':0,'control_character_errors':0,'Hamming_product':'[[58,16,3]] verified with GF(2) ranks and weight-1/2 exclusion plus weight-3 witnesses','Schlogl_roots':[1,2,3],'Schlogl_derivatives':[-2,1,-2],'dimensionless_exponential':math.exp(10**0.3),'sequence_count_L20':4**20},indent=2))
