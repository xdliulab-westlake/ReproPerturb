"""Generate all new unified-parameter reports, tables and pathway visualizations."""
from pathlib import Path
import json,csv,html,importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from hypo_release.paths import ROOT as R
from hypo_release.unified import examples

def read(p):return json.loads(Path(p).read_text())
def makepdf(path,lines,figures):
 style=ParagraphStyle('bilingual',fontName='STSong-Light',fontSize=10,leading=15)
 story=[]
 for line in lines:story += [Paragraph(html.escape(line),style),Spacer(1,8)]
 for title,p in figures:story += [PageBreak(),Paragraph(html.escape(title),style),Image(str(p),width=510,height=300)]
 SimpleDocTemplate(str(path),leftMargin=40,rightMargin=40).build(story)
def main():
 out=R/'reports';out.mkdir(exist_ok=True);figdir=out/'figures';figdir.mkdir(exist_ok=True)
 results=read(R/'RESULTS.json');c=read(R/'SHARED_CONFIG.json');u=read(R/'SHARED_UNKNOWN_K.json')
 pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
 shared=['17-pathway unified-parameter rerun / 17通路统一参数重跑报告','2026-10-27',
 '本版三个seed共用同一响应特征方案、核超参数、身份/方向混合系数，以及未知K的先验权重和阈值。仅训练得到的权重、训练特征标准化统计量和预测数值允许不同。',
 'All seeds share feature choice, kernel hyperparameters, mixing coefficients and unknown-K calibration. Fitted weights, training normalization statistics and predictions remain seed-specific.',
 '前向Flow架构及训练配置原本一致，复用原epoch150权重，未重新训练Flow。此次重新计算响应特征、共同训练侧OOF选择、重新拟合关系模型、重新执行6目标/11条协议验证、前向验证和组合实验。',
 'Frozen epoch150 Flow checkpoints are reused. Fresh response extraction, pooled training-OOF configuration selection, relation refits, six-target/panel validation, forward validation and combination audits are rerun. No new full Flow training is claimed.',
 'Common relation configuration: '+json.dumps(c,ensure_ascii=False),
 'Common unknown-K configuration: '+json.dumps(u,ensure_ascii=False),
 '选择规则：同一组合分组五折，随机种子904；关系配置最大化三seed训练OOF平均(signed F1 +0.2协议完全匹配率)。未知K最大化训练OOF平均(signed F1 +0.15宏平均完全匹配率)。并列时固定取枚举中先出现者。测试结果不参与参数重选。',
 'Same grouped folds across seeds; group RNG904. Relation selection maximizes mean training-OOF utility. Unknown-K calibration uses only training-OOF scores and fold-specific frequency priors. No holdout-result-based reselection.',
 '数据边界：Stage2划分146/21/14；关系弱监督池与Stage2划分不是同一概念，保留原77条监督和原64个双药背景。旧监督池相对后续划分存在重叠问题；Stage1未经完整holdout审计；本任务和面板曾被反复查看。因此仍是开发性评估，不是新盲测，也不能因统一参数而宣称解决了历史泄露问题。',
 'Historical weak-label/context pools are retained, not silently redesigned. Reused Stage1 and prior panel exposure remain limitations. Shared hyperparameters improve comparability but do not establish independent blind validation.',
 '六目标采用统一的task-aware规则：共同关系模型给出身份排序，直接Flow边际提供方向；11协议采用共同signed-hybrid方向。不同任务的规则不同，但每项任务跨seed完全一致。',
 'Hypoblast uses the same task-aware identity/direct-Flow-sign rule for every seed; the panel uses shared signed-hybrid signs. These are different tasks, not per-seed model choices.']
 known=list(shared);unknown=list(shared)
 for s in ['43','44','45']:
  r=results[s];k=r['known_k'];v=r['unknown_k'];six=r['hypoblast']
  known.append(f"Seed{s}: Hypoblast signed Hits@6 {six['signed_hits_at_6']}/6; panel celltypes {k['celltypes_passed']}/6; signed hits {k['signed']}/{k['denominator']}; selected {k['predicted_total']}. Six selected: "+', '.join(x['pathway']+x['direction'] for x in six['selected']))
  unknown.append(f"Seed{s}: panel celltypes {v['any_protocol_celltypes']}/6; signed hits {v['signed']}/{v['truth_total']}; selected {v['predicted_total']}; signed F1 {v['signed_f1']:.4f}.")
 known.append('Mean panel celltypes: '+str(np.mean([results[s]['known_k']['celltypes_passed'] for s in results]))+'/6')
 unknown.append('Mean panel celltypes: '+str(np.mean([results[s]['unknown_k']['any_protocol_celltypes'] for s in results]))+'/6')
 for lines in [known,unknown]:
  lines.append('细胞类型通过定义：同类型的至少一个预测集合，完全匹配同类型至少一条参考协议，包含方向。不同于“每个样本自身协议完全正确”。具体逐样本匹配另存CSV。 / Celltype pass allows matching any reference of that celltype; own-protocol exact match is separately reported.')
 tr,te,paths,groups=examples(43)
 truth=np.array([[1 if e.truth.get(p)=='+' else -1 if e.truth.get(p)=='-' else 0 for p in paths] for e in te])
 figures={'known':[],'unknown':[]}
 allrows=[]
 for kind in ['known_k','unknown_k']:
  for s in ['43','44','45']:
   rr=results[s][kind]['rows'];pred=np.array([[1 if r['prediction'].get(p)=='+' else -1 if r['prediction'].get(p)=='-' else 0 for p in paths] for r in rr])
   fig,axs=plt.subplots(1,2,figsize=(12,6),sharey=True)
   for ax,a,title in zip(axs,[truth,pred],['Reference protocol','Prediction']):
    ax.imshow(a,cmap=ListedColormap(['#377eb8','#eeeeee','#e41a1c']),vmin=-1,vmax=1,aspect='auto');ax.set_title(title)
    ax.set_xticks(range(17),paths,rotation=65,ha='right',fontsize=7);ax.set_yticks(range(11),[e.sample_id+' '+e.target_celltype for e in te],fontsize=7)
   fig.suptitle(f'{kind} seed{s} | red:+ blue:- gray:not selected');fig.tight_layout();p=figdir/f'{kind}_seed{s}.png';fig.savefig(p,dpi=150);plt.close(fig);figures['known' if kind=='known_k' else 'unknown'].append((kind+' seed'+s,p))
   for r in rr:
    allrows.append(dict(variant=kind,seed=s,**r))
    line=f"Seed{s} {r['sample']} / {r['celltype']}: {r['signed']}/{r['truth_k']} signed hits, predicted K={r['predicted_k']}, own-protocol exact={r['exact']}. "+', '.join(p+d for p,d in r['prediction'].items())
    (known if kind=='known_k' else unknown).append(line)
 fig,ax=plt.subplots(figsize=(10,6));a=np.array([[next(x['score'] for x in results[s]['hypoblast']['rankings'] if x['pathway']==p) for s in ['43','44','45']] for p in paths]);im=ax.imshow(a,aspect='auto',cmap='viridis');ax.set_yticks(range(17),paths);ax.set_xticks(range(3),['43','44','45']);ax.set_title('All 17 pathway identity scores | signed labels, * selected')
 for j,s in enumerate(['43','44','45']):
  selected={x['pathway'] for x in results[s]['hypoblast']['selected']}
  for i,p in enumerate(paths):ax.text(j,i,next(x['direction'] for x in results[s]['hypoblast']['rankings'] if x['pathway']==p)+('*' if p in selected else ''),ha='center',va='center',color='white')
 fig.colorbar(im,ax=ax);fig.tight_layout();p=figdir/'hypoblast_all_pathways.png';fig.savefig(p,dpi=160);plt.close(fig);figures['known'].insert(0,('Hypoblast all17 pathways',p))
 fields=['variant','seed','sample','celltype','truth_k','predicted_k','signed','identity','exact','prediction']
 with (out/'PANEL_RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for row in allrows:row=dict(row,prediction=json.dumps(row['prediction']));w.writerow(row)
 combo=['统一参数版本：组合递增实验 / Unified release combination-response audit',
 'Both arms rerun using frozen seed-matched Flow. The reference-combination arm does not depend on relation-head hyperparameters; unchanged predictions are expected when weights/inputs are unchanged. Unknown-K arm uses the newly shared-config selected sets.',
 '所有参考子集；所有单项额外添加；最多32对固定随机抽样额外添加。容差0.001比例。平均曲线峰值不代表所有子集/额外添加比较均通过。',
 'All subsets, all eligible single additions, up to32 sampled pairs; tolerance0.001 fraction. Curves are means across combinations of each size, not an optimally chosen addition sequence.']
 combo_figs=[]
 # Recompute summaries/plots from newly executed known-query records.
 spec=importlib.util.spec_from_file_location('known_report',R/'experiments/combination_growth/code/report_known_k.py');mod=importlib.util.module_from_spec(spec)
 import os
 os.environ['HYPO_COMBO_OUTPUT']=str(R/'experiments/combination_growth/known_k');spec.loader.exec_module(mod);mod.main()
 summaries=read(R/'experiments/combination_growth/known_k/SUMMARY.json')
 panel=[x for x in summaries if x['seed']=='mean' and not x['case'].startswith('Hypoblast')]
 combo.append(f"Reference drug protocols, three-seed mean: mean-curve peak at full {sum(x['full_is_mean_curve_peak'] for x in panel)}/11; full >= all subsets {sum(x['full_ge_all_subsets'] for x in panel)}/11; all checks {sum(x['all_three_checks'] for x in panel)}/11.")
 for x in summaries:
  if x['seed']=='mean':combo_figs.append((x['case'],R/f"experiments/combination_growth/known_k/figures/{x['case']}.png"))
 rows=read(R/'experiments/combination_growth/unknown_k/SUMMARY.json')
 for s in [43,44,45]:
  rr=[r for r in rows if r['seed']==s];combo.append(f"Unknown-K seed{s}: mean-curve peak at selected K {sum(r['mean_peak_at_selected_k'] for r in rr)}/11; >=all subsets {sum(r['full_ge_all_subsets'] for r in rr)}/11; no single gain {sum(r['no_single_gain'] for r in rr)}/11; no sampled-pair gain {sum(r['no_pair_gain'] for r in rr)}/11.")
 for sid in dict.fromkeys(r['sample'] for r in rows):
  fig,axs=plt.subplots(1,3,figsize=(12,4))
  for ax,s in zip(axs,[43,44,45]):
   row=next(r for r in rows if r['sample']==sid and r['seed']==s);ks=sorted(int(k) for k in row['curve']);ax.plot(ks,[100*row['curve'][str(k)] for k in ks],'o-');ax.axvline(row['predicted_k'],ls='--',c='black');ax.set(title=f"{sid} seed{s}, K={row['predicted_k']}",xlabel='Active pathway count',ylabel='Target fraction (%)')
  fig.tight_layout();p=figdir/f'unknown_combo_{sid}.png';fig.savefig(p,dpi=150);plt.close(fig);combo_figs.append(('Unknown-K '+sid,p))
 for name,lines,figs in [('KNOWN_K',known,figures['known']),('UNKNOWN_K',unknown,figures['unknown']),('COMBINATION',combo,combo_figs),('FULL',known+unknown[len(shared):]+combo,figures['known']+figures['unknown']+combo_figs)]:
  (out/f'{name}_REPORT_BILINGUAL.txt').write_text('\n\n'.join(lines));makepdf(out/f'{name}_REPORT_BILINGUAL.pdf',lines,figs)
 print('REPORTS COMPLETE',out,flush=True)
if __name__=='__main__':main()
