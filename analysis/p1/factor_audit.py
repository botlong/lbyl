"""Independent P1 audit and fixed-seed name/role factorial. Run from m0."""
import os,sys,json,hashlib,collections,itertools
from pathlib import Path
HERE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(HERE));os.chdir(HERE)
import torch
torch.set_num_threads(8)
import train_gphi as tg
from stats_util import input_key,mean_ci
OUT=HERE/'results/neucom_p1/analysis';OUT.mkdir(parents=True,exist_ok=True)
def rows(d):return [json.loads(l) for l in (HERE/d/'gphi_data.jsonl').read_text().splitlines()]
def blob(d,f='feats.pt'):return torch.load(HERE/d/f,weights_only=False,map_location='cpu')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def overlap(tr,te,fs):
 s={input_key(fs[r['crop']]) for r in tr};return sum(input_key(fs[r['crop']]) in s for r in te)
audit={'datasets':{},'splits':{},'code_hashes':{p.name:digest(p) for p in HERE.glob('*.py')}}
for d in ['gphi_data_55','gphi_data','gphi_data_realistic_v2','gphi_data_crossmodal','gphi_data_class2','gphi_data_susbench','gphi_data_susbench_rawaria','gphi_data_susbench_dest']:
 rs=rows(d);b=blob(d);fs=b['feats'];missing=[r['crop'] for r in rs if r['crop'] not in fs]
 bad=[k for k,f in fs.items() if not all(torch.isfinite(f[c]).all().item() for c in ['vis','txt'])]
 crop_missing=[r['crop'] for r in rs if not (HERE/d/'crops'/r['crop']).exists()]
 conflicting=collections.defaultdict(set)
 for r in rs:conflicting[input_key(fs[r['crop']])].add(r['label'])
 info={'n':len(rs),'positive':sum(r['label'] for r in rs),'unique_pairs':len(conflicting),'conflicting_pairs':sum(len(v)>1 for v in conflicting.values()),'missing_features':missing,'nonfinite':bad,'missing_crops':crop_missing,'raw_nonempty':sum(bool(r.get('aria','').strip()) for r in rs),'computed_nonempty':sum(bool(r.get('ax_name','').strip()) for r in rs),'name_equals_visible_exact':sum(r.get('ax_name','').strip()==r.get('visible_text','').strip() for r in rs),'name_equals_visible_casefold':sum(r.get('ax_name','').strip().lower()==r.get('visible_text','').strip().lower() for r in rs),'metadata_sha256':digest(HERE/d/'gphi_data.jsonl'),'features_sha256':digest(HERE/d/'feats.pt'),'model':b.get('model'),'struct':b.get('struct')}
 audit['datasets'][d]=info
 print('DATASET',d,json.dumps(info),flush=True)
 if d=='gphi_data_55':
  audit['splits']['env_loto']={t:overlap([r for r in rs if r['template']!=t],[r for r in rs if r['template']==t],fs) for t in sorted({r['template'] for r in rs})}
  audit['splits']['env_level']=overlap([r for r in rs if r['level'] in [0,1]],[r for r in rs if r['level']==2],fs)
 if d=='gphi_data_realistic_v2':
  import crossmodal_eval as cm
  for ff in ['feats.pt','feats_hidden.pt','feats_internvl.pt']:
   bf=blob(d,ff);audit['splits'][d+'/'+ff]={n:{'train':len(tr),'test':len(te),'overlap':overlap(tr,te,bf['feats'])} for n,tr,te in cm.split_v2(rs)}
  types=sorted({r['template'] for r in rs})
  audit['splits']['sweep364_overlap']=[{'held':list(t),'overlap':overlap(*cm.holdout(rs,set(t)),fs)} for t in itertools.combinations(types,3)]
 if d=='gphi_data_crossmodal':
  import crossmodal_eval as cm
  tr,te=cm.split_pair(rs);audit['splits']['grid_pair']={'train':len(tr),'test':len(te),'overlap':overlap(tr,te,fs),'test_positive':sum(r['label'] for r in te),'test_majority_accuracy':max(sum(r['label'] for r in te),sum(not r['label'] for r in te))/len(te)}
(OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
print('AUDIT_DONE',flush=True)
# Embed-mode text features require only the frozen token embedding tensor, on CPU.
from transformers import AutoTokenizer
from safetensors import safe_open
model=HERE.parent/'models/Qwen2.5-VL-3B-Instruct';idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
key=next(k for k in idx if k.endswith('embed_tokens.weight'))
with safe_open(model/idx[key],framework='pt',device='cpu') as f:weight=f.get_tensor(key)
tok=AutoTokenizer.from_pretrained(model);cache={}
def enc(s):
 if s not in cache:cache[s]=weight[tok(s,return_tensors='pt').input_ids[0]].mean(0).float()
 return cache[s]
rs=rows('gphi_data_55');b=blob('gphi_data_55');vd,td=b['vis_dim'],b['txt_dim'];tpls=sorted({r['template'] for r in rs})
tg.NO_NUM=True;tg.ZERO_VIS=tg.ZERO_TXT=False
results={'protocol':'same visual inputs; raw/computed name x tag/AX role; numeric-free; 8 seeds; fold seed=1000*run+template_index; 200 epochs; threshold 0.5','configs':{},'template_metadata':{t:{'n':len(rr:=[r for r in rs if r['template']==t]),'empty_raw':sum(not r.get('aria','').strip() for r in rr),'changed_name':sum(r.get('aria','').strip()!=r.get('ax_name','').strip() for r in rr),'changed_role':sum(r.get('role','')!=r.get('ax_role','') for r in rr)} for t in tpls}}
for name,nk,rk in [('raw_tag','aria','role'),('computed_tag','ax_name','role'),('raw_axrole','aria','ax_role'),('computed_axrole','ax_name','ax_role')]:
 fs={r['crop']:{'vis':b['feats'][r['crop']]['vis'],'txt':enc((r.get(nk) or ' ')+' '+(r.get(rk) or ''))} for r in rs}
 if name=='computed_axrole':
  delta=max((fs[r['crop']]['txt']-b['feats'][r['crop']]['txt']).abs().max().item() for r in rs);print('EMBED_REPRO_MAX_ABS',delta,flush=True);assert delta==0
 cfg={'seeds':[],'per_template':{t:[] for t in tpls}}
 for s in range(8):
  ms=[]
  for i,t in enumerate(tpls):
   tr=[r for r in rs if r['template']!=t];te=[r for r in rs if r['template']==t]
   m=tg.train_head(tr,vd,td,fs,seed=1000*s+i);metric,_=tg.evaluate(m,te,fs);ms.append(metric);cfg['per_template'][t].append(metric)
  mean={k:sum(m[k] for m in ms)/len(ms) for k in ['recall','fpr','acc']};cfg['seeds'].append(mean);print('FACTOR',name,s,json.dumps(mean),flush=True)
  results['configs'][name]=cfg;(OUT/'factor_results.partial.json').write_text(json.dumps(results,indent=2))
 cfg['summary']={k:dict(zip(['mean','t95_half','sd'],mean_ci([m[k] for m in cfg['seeds']]))) for k in ['recall','fpr','acc']}
 print('FACTOR_SUMMARY',name,json.dumps(cfg['summary']),flush=True)
(OUT/'factor_results.json').write_text(json.dumps(results,indent=2));print('FACTOR_DONE',flush=True)
