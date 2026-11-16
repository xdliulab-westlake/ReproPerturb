from pathlib import Path
import json,re,py_compile
import numpy as np
I=Path('/liaozizhuo/hypoblast/benchmarks/hypo-independent-intervention-20261112');B=I/'perturbgen_task';run=B/'runs/perturbgen_seed43';cache=run/'cache';shards=cache/'val_shards';shards.mkdir(exist_ok=True)
log=(B/'logs/seed43_count.log').read_text();done=re.findall(r'CACHE val_generated (\S+) (\d+) (\d+)',log);a=np.load(cache/'val_generated.npy',mmap_mode='r');start=0
for sid,end,_ in done:
 end=int(end);np.save(shards/(sid+'.npy'),np.array(a[start:end]));start=end
s=(B/'code/finish_perturbgen.py').read_text().replace("ap.add_argument('--smoke',action='store_true');a=ap.parse_args()","ap.add_argument('--smoke',action='store_true');ap.add_argument('--cache-worker',type=int,default=-1);a=ap.parse_args()")
s=s.replace("    write(run/'count_manifest.json',", "    write(run/('parallel_count_manifest.json' if a.cache_worker<0 else f'cache_worker{a.cache_worker}_manifest.json'),")
anchor="        n=sum(map(len,chosen.values()));"
insert="""        if label=='val_generated' and a.cache_worker<0:
            shardroot=cache/'val_shards'
            for sid in ids:
                while not (shardroot/(sid+'.npy')).exists():time.sleep(10)
            offsets={};allix=[];offset=0;n=sum(map(len,chosen.values()))
            out=np.lib.format.open_memmap(f,mode='w+',dtype=np.float32,shape=(n,768))
            for sid in ids:
                z=np.load(shardroot/(sid+'.npy'),mmap_mode='r');ix=chosen[sid]
                assert z.shape==(len(ix),768) and np.isfinite(z).all()
                out[offset:offset+len(ix)]=z;offsets[sid]=[offset,offset+len(ix)];offset+=len(ix);allix.extend(ix.tolist())
            out.flush();meta=dict(mask_checkpoint_sha256=mask_sha,shape=list(out.shape),offsets=offsets,target_tokens_used=False,sampling_seed=None)
            np.save(cache/(label+'_rows.npy'),np.array(allix));write(info,meta)
            return out,meta
"""
s=s.replace(anchor,insert+anchor)
anchor="    train,trainmeta=build_cache('train_teacher'"
pos=s.index(anchor)
worker="""    if a.cache_worker>=0:
        for j,sid in enumerate(cfg['splits']['val']):
            dest=cache/'val_shards'/(sid+'.npy')
            if j%2!=a.cache_worker or dest.exists():continue
            z,meta=build_cache('parallel_val_'+sid,[sid])
            # Atomic promotion; model seed and per-condition generation seed unchanged.
            import os
            tmp=dest.with_suffix('.partial.npy');np.save(tmp,np.asarray(z));os.replace(tmp,dest)
            print('SHARD_READY',sid,flush=True)
        print('WORKER_COMPLETE',a.cache_worker,flush=True);return

"""
s=s[:pos]+worker+s[pos:];p=B/'code/finish_perturbgen_parallel_cache.py';p.write_text(s);py_compile.compile(str(p),doraise=True)
(I/'parallel_cache_preserved.json').write_text(json.dumps({'conditions':[x[0] for x in done],'cells':start,'source':'seed43_count.log; flushed per-condition cache','workers':2,'same_condition_rng_seed':True},indent=2));print('PRESERVED',len(done),start)
