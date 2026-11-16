"""Rank all 34 signed relation scores directly, without collapsing directions."""
import argparse, csv
from collections import defaultdict

def rank(rows):
    grouped=defaultdict(dict)
    for row in rows:
        key=(row['celltype'],row['pathway'],row['direction'])
        seed=int(row['seed'])
        if seed in grouped[key]: raise ValueError('Duplicate seed/candidate')
        grouped[key][seed]=float(row['score'])
    cells=defaultdict(list)
    for (cell,path,direction),scores in grouped.items():
        if set(scores)!={43,44,45}: raise ValueError('Expected seeds 43/44/45')
        cells[cell].append(dict(celltype=cell,pathway=path,direction=direction,score=sum(scores.values())/3))
    result=[]
    for cell,items in sorted(cells.items()):
        if len(items)!=34 or len({i['pathway'] for i in items})!=17 or any({i['direction'] for i in items if i['pathway']==p}!={'+','-'} for p in {i['pathway'] for i in items}): raise ValueError('Expected 17 pathways x 2 directions')
        for item in sorted(items,key=lambda x:(-x['score'],x['pathway'],x['direction'])):
            item['rank']=1+sum(x['score']>item['score'] for x in items); result.append(item)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with open(a.scores) as f: rows=rank(csv.DictReader(f))
    with open(a.output,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['celltype','rank','pathway','direction','score']);w.writeheader();w.writerows(rows)
