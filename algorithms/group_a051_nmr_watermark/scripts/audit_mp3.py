import json,os,time
from pathlib import Path
import numpy as np
import soundfile as sf
root=Path('outputs/mp3-audit');root.mkdir(exist_ok=True)
paths=sorted({r['path'] for split in ('train','validation') for r in json.loads(Path(f'data/manifests/full/{split}.json').read_text())})
errors=[];start=time.monotonic()
with (root/'decode-stderr.log').open('wb',buffering=0) as log:
 saved=os.dup(2);os.dup2(log.fileno(),2)
 try:
  for i,p in enumerate(paths):
   os.write(2,('\nFILE '+p+'\n').encode('utf-8'))
   try:
    a,sr=sf.read(p,dtype='float32',always_2d=True)
    if not len(a) or not np.isfinite(a).all():errors.append([p,'invalid samples'])
   except Exception as e:errors.append([p,str(e)])
   if i%500==0:print(i,len(paths),round(time.monotonic()-start),flush=True)
 finally:os.dup2(saved,2);os.close(saved)
text=(root/'decode-stderr.log').read_text(encoding='utf-8',errors='replace');bad=[]
for block in text.split('\nFILE ')[1:]:
 lines=block.splitlines()
 if any(line.strip() for line in lines[1:]):bad.append({'path':lines[0],'messages':lines[1:]})
result=dict(files=len(paths),seconds=time.monotonic()-start,warnings=bad,errors=errors)
(root/'scan.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
