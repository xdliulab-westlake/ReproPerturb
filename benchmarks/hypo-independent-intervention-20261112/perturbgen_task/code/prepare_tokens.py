"""Stream the existing log1p(CP10k) h5ad into fixed official gene-rank tokens.

Only train/validation target cells and allowed control source cells are read.
Test target rows stay all-zero and cannot enter the training dataset.
"""
import hashlib,json,pickle,time
from pathlib import Path
import h5py
import numpy as np
from prepare_composition import ROOT,OLD,read,write,sha


def main():
    cfg=read(OLD/'benchmark.json');rows=dict(np.load(OLD/'data/sample_rows_train_val.npz'))
    vendor=ROOT/'vendor/Perturbgen';pp=vendor/'perturbgen/pp'
    token=pickle.loads((pp/'token_dict_gftokens_gc95M.pkl').read_bytes())
    med=pickle.loads((pp/'gene_median_dict_gftokens_gc95M.pkl').read_bytes())
    aliases=pickle.loads((pp/'ensembl_mapping_dict_gc95M.pkl').read_bytes())
    source=ROOT.parents[1]/'data/hypo/hypo_final_20260313.h5ad'
    out=ROOT/'data';out.mkdir(exist_ok=True)
    with h5py.File(source,'r') as f:
        names=f['var/_index'].asstr()[:];n=int(f['X'].attrs['shape'][0]);genes=[]
        for name in names:
            e=aliases.get(name,name);genes.append(e)
        geneid=np.array([token.get(e,-1) for e in genes]);medians=np.ones(max(token.values())+1)
        for e,v in med.items():medians[token[e]]=v
        wanted=np.zeros(n,dtype=bool)
        for sid in cfg['splits']['train']+cfg['splits']['val']+sorted({r['control_sample_id'] for r in cfg['cohort']}):wanted[rows[sid]]=True
        first=next(iter(rows.values()))[0]
        assert f['obs/_index'].asstr()[first].endswith('-'+next(iter(rows))), 'Cell index identity mismatch'
        dest=out/'tokens_train_val_source.npy';done=ROOT/'audit/tokens_complete.json'
        if done.exists():print('TOKENS_ALREADY_COMPLETE',flush=True);return
        tokens=np.lib.format.open_memmap(dest,mode='w+',dtype=np.uint16,shape=(n,4096))
        ptr=f['X/indptr'][:];lengths=np.zeros(n,dtype=np.uint16);scalereport=[];start_time=time.time()
        for begin in range(0,n,2048):
            end=min(n,begin+2048)
            if not wanted[begin:end].any():continue
            a,b=int(ptr[begin]),int(ptr[end]);ind=f['X/indices'][a:b];values=f['X/data'][a:b]
            if (values<0).any() or not np.isfinite(values).all():raise ValueError('Invalid lognorm values')
            values=np.expm1(values.astype(np.float64))
            for i in range(begin,end):
                if not wanted[i]:continue
                lo,hi=int(ptr[i]-a),int(ptr[i+1]-a);ids=geneid[ind[lo:hi]];v=values[lo:hi]
                if len(scalereport)<50:scalereport.append(float(v.sum()))
                good=ids>=4;v=np.bincount(ids[good],weights=v[good],minlength=len(medians))/medians
                expressed=np.flatnonzero(v>0);order=expressed[np.argsort(-v[expressed],kind='stable')][:4094]
                seq=np.r_[2,order,3].astype(np.uint16);tokens[i,:len(seq)]=seq;lengths[i]=len(seq)
            if begin%65536==0:
                tokens.flush();write(ROOT/'audit/tokens_progress.json',dict(status='tokenizing',rows_scanned=end,total_rows=n,elapsed_seconds=time.time()-start_time));print('TOKEN_ROWS',end,n,flush=True)
        tokens.flush();np.save(out/'token_lengths_train_val_source.npy',lengths)
        # CP10k scaling is a positive within-cell scalar and does not alter ranks.
        if not np.allclose(scalereport,10000,rtol=.01):raise ValueError(('Unexpected h5ad normalization',scalereport))
        allrows=dict(np.load(OLD/'data/sample_rows_all.npz'))
        assert all(not lengths[allrows[s]].any() for s in cfg['splits']['test'])
        info=dict(status='complete',shape=list(tokens.shape),dtype=str(tokens.dtype),n_input_cells=int(wanted.sum()),test_target_tokens=0,
          gene_columns=len(names),mapped_columns=int((geneid>=4).sum()),vocabulary_size=len(token),source=str(source),source_size=source.stat().st_size,
          rank_normalization='expm1(log1p CP10k), collapse aliases, divide by fixed official gene medians',source_sequence_length=4096,target_sequence_max=2048,
          tie_rule='stable ascending official token ID',official_token_sha256=sha(pp/'token_dict_gftokens_gc95M.pkl'),official_median_sha256=sha(pp/'gene_median_dict_gftokens_gc95M.pkl'),seconds=time.time()-start_time)
        write(done,info);print('TOKENS_COMPLETE',info,flush=True)


if __name__=='__main__':main()
