# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
import math, re, json, csv, sys
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

@dataclass
class Netlist:
    name: str
    vertices: list[str]
    nets: dict[str, list[int]]
    primary_inputs: set[str]
    primary_outputs: set[str]
    io_nets: set[str]


def logical_lines(path: Path):
    buf=''
    for raw in path.read_text(errors='ignore').splitlines():
        line=raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.endswith('\\'):
            buf += line[:-1]+' '
            continue
        line=buf+line; buf=''
        yield line
    if buf: yield buf


def parse_blif(path: Path) -> Netlist:
    lines=list(logical_lines(path))
    inputs=[]; outputs=[]; gates=[]
    i=0
    while i<len(lines):
        toks=lines[i].split()
        if not toks: i+=1; continue
        if toks[0]=='.inputs': inputs.extend(toks[1:])
        elif toks[0]=='.outputs': outputs.extend(toks[1:])
        elif toks[0]=='.names':
            if len(toks)>=2:
                ins=toks[1:-1]; out=toks[-1]
                gates.append((out,ins))
            # following truth table lines ignored
        i+=1
    vertices=[]; idx={}
    def add(v):
        if v not in idx:
            idx[v]=len(vertices); vertices.append(v)
        return idx[v]
    # Explicit input and output terminal vertices
    for x in inputs: add('PI:'+x)
    for out,_ in gates: add('G:'+out)
    for y in outputs: add('PO:'+y)
    driver={x:'PI:'+x for x in inputs}
    for out,_ in gates: driver[out]='G:'+out
    consumers=defaultdict(list)
    for out,ins in gates:
        gv='G:'+out
        for s in ins: consumers[s].append(gv)
    for y in outputs: consumers[y].append('PO:'+y)
    nets={}
    allsignals=set(driver)|set(consumers)
    for s in sorted(allsignals):
        pins=[]
        if s in driver: pins.append(add(driver[s]))
        for v in consumers.get(s,[]): pins.append(add(v))
        pins=sorted(set(pins))
        if len(pins)>=2: nets[s]=pins
    io_nets=set(inputs)|set(outputs)
    return Netlist(path.stem,vertices,nets,set(inputs),set(outputs),io_nets)


def projection_adjacency(nl: Netlist) -> sp.csr_matrix:
    n=len(nl.vertices); m=len(nl.nets)
    rows=[]; cols=[]; data=[]
    for j,pins in enumerate(nl.nets.values()):
        d=len(pins)
        w=1.0/math.sqrt(max(1,d-1))
        for v in pins:
            rows.append(v); cols.append(j); data.append(w)
    H=sp.csr_matrix((data,(rows,cols)),shape=(n,m))
    A=(H@H.T).tocsr()
    A.setdiag(0); A.eliminate_zeros()
    # Sparse matrix products may emit equivalent CSR rows with a
    # platform-dependent column order.  RCM uses neighbor order to break
    # degree ties, so canonicalize the representation before ordering.
    A.sum_duplicates(); A.sort_indices()
    return A


def spectral_split(A: sp.csr_matrix, verts: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    k=len(verts)
    if k<=2:
        mid=k//2; return verts[:mid],verts[mid:]
    sub=A[verts][:,verts].astype(float)
    deg=np.asarray(sub.sum(axis=1)).ravel()
    # disconnected/isolated fallback: deterministic degree+index order
    if np.count_nonzero(deg)>1 and sub.nnz>0:
        d_inv=np.zeros_like(deg)
        nz=deg>1e-12; d_inv[nz]=1/np.sqrt(deg[nz])
        L=sp.eye(k,format='csr')-sp.diags(d_inv)@sub@sp.diags(d_inv)
        try:
            if k<=12:
                vals,vecs=np.linalg.eigh(L.toarray())
                v=vecs[:,1] if k>1 else np.arange(k)
            else:
                v0=np.sin(np.arange(k)+1.2345)
                vals,vecs=spla.eigsh(L,k=2,which='SM',v0=v0,tol=1e-5,maxiter=max(1000,10*k))
                order=np.argsort(vals); v=vecs[:,order[1]]
            order=np.lexsort((verts,v))
        except Exception:
            order=np.lexsort((verts,-deg))
    else:
        order=np.argsort(verts)
    mid=k//2
    left=verts[order[:mid]]; right=verts[order[mid:]]
    if len(left)==0 or len(right)==0:
        ord2=np.sort(verts); mid=k//2; left,right=ord2[:mid],ord2[mid:]
    return left,right


def recursive_hierarchy(A: sp.csr_matrix, n: int, min_size=8):
    levels=[]
    modules=[np.arange(n,dtype=int)]
    levels.append([m.copy() for m in modules])
    while max(map(len,modules))>min_size:
        new=[]
        for mod in modules:
            if len(mod)<=min_size:
                new.append(mod)
            else:
                l,r=spectral_split(A,mod)
                new.extend([l,r])
        modules=new
        levels.append([m.copy() for m in modules])
        if len(levels)>32: break
    return levels



def rcm_hierarchy(A: sp.csr_matrix, n: int, min_size=8):
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    order=reverse_cuthill_mckee(A, symmetric_mode=True)
    levels=[]
    m=1
    while True:
        blocks=[b.astype(int) for b in np.array_split(order,m) if len(b)>0]
        levels.append(blocks)
        if max(map(len,blocks))<=min_size: break
        m*=2
        if m>n: break
    return levels


def recursive_rcm_hierarchy(A: sp.csr_matrix, n: int, min_size=8):
    """Deterministic recursive bisection using a fresh local RCM order per module."""
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    levels=[]
    modules=[np.arange(n,dtype=int)]
    levels.append([m.copy() for m in modules])
    while max(map(len,modules))>min_size:
        new=[]
        for mod in modules:
            if len(mod)<=min_size:
                new.append(mod)
                continue
            if len(mod)>1024:
                sub=A[mod][:,mod].tocsr()
                sub.sum_duplicates(); sub.sort_indices()
                order=reverse_cuthill_mckee(sub, symmetric_mode=True)
                ordered=mod[np.asarray(order,dtype=int)]
            else:
                # Preserve the inherited local order below the coarsening threshold.
                ordered=mod
            mid=len(ordered)//2
            new.extend([ordered[:mid],ordered[mid:]])
        modules=new
        levels.append([m.copy() for m in modules])
        if len(levels)>32:
            break
    return levels

def fanout_class(d):
    if d==2: return '2-pin'
    if d<=8: return '3-8 pin'
    if d<=32: return '9-32 pin'
    return '>32 pin'


def level_metrics(nl:Netlist, levels):
    rec=[]
    for depth,mods in enumerate(levels):
        assign=np.empty(len(nl.vertices),dtype=int)
        sizes=[]
        for j,m in enumerate(mods):
            assign[m]=j; sizes.append(len(m))
        per_mod_total=np.zeros(len(mods),dtype=int)
        per_mod_internal=np.zeros(len(mods),dtype=int)
        class_counts={c:np.zeros(len(mods),dtype=int) for c in ['2-pin','3-8 pin','9-32 pin','>32 pin']}
        io_counts=np.zeros(len(mods),dtype=int)
        class_aggregate=defaultdict(int)
        for name,pins in nl.nets.items():
            touched=np.unique(assign[np.array(pins,dtype=int)])
            q=len(touched)
            if q>=2:
                per_mod_total[touched]+=1
                cls=fanout_class(len(pins)); class_counts[cls][touched]+=1
                class_aggregate[cls]+=q
                if name in nl.io_nets: io_counts[touched]+=1
                else: per_mod_internal[touched]+=1
            # pad-inclusive convention: external net counts at every module it touches
            if name in nl.io_nets:
                # add if not already crossing; this makes whole-root show external terminals
                if q==1:
                    per_mod_total[touched]+=1; io_counts[touched]+=1
        m=len(mods); G=len(nl.vertices); g=G/m
        row={
            'name':nl.name,'depth':depth,'modules':m,'G':G,'g_nominal':g,
            'size_min':min(sizes),'size_max':max(sizes),'size_mean':float(np.mean(sizes)),
            'T_level_mean':float(np.mean(per_mod_total)),
            'T_level_mean_internal':float(np.mean(per_mod_internal)),
            'logT_scatter_mean':float(np.mean(np.log(np.maximum(per_mod_total,1)))),
            'logT_scatter_internal':float(np.mean(np.log(np.maximum(per_mod_internal,1)))),
            'pad_io_mean':float(np.mean(io_counts)),
        }
        for cls,arr in class_counts.items(): row['T_'+cls]=float(np.mean(arr))
        rec.append(row)
    return rec


def ols_slope(xs,ys):
    x=np.asarray(xs,float); y=np.asarray(ys,float)
    xm=x.mean(); ym=y.mean(); den=np.sum((x-xm)**2)
    return float(np.sum((x-xm)*(y-ym))/den) if den>0 else float('nan')


def summarize(rows):
    # Exclude root, tiny levels; use 16 <= nominal g <= G/4, positive T
    valid=[r for r in rows if 16<=r['g_nominal']<=r['G']/4 and r['T_level_mean_internal']>0]
    x=[math.log(r['g_nominal']) for r in valid]
    y_level=[math.log(r['T_level_mean_internal']) for r in valid]
    y_scatter=[r['logT_scatter_internal'] for r in valid]
    p_level=ols_slope(x,y_level) if len(x)>=2 else float('nan')
    p_scatter=ols_slope(x,y_scatter) if len(x)>=2 else float('nan')
    # mode slopes
    mode_slopes={}
    for cls in ['2-pin','3-8 pin','9-32 pin','>32 pin']:
        vv=[r for r in valid if r['T_'+cls]>0]
        if len(vv)>=2:
            mode_slopes[cls]=ols_slope([math.log(r['g_nominal']) for r in vv],[math.log(r['T_'+cls]) for r in vv])
        else: mode_slopes[cls]=float('nan')
    # Jensen gap average over valid levels
    gaps=[]
    for r in valid:
        # log mean T - mean log max(T,1), approximate from stored data
        gaps.append(math.log(max(r['T_level_mean_internal'],1e-12))-r['logT_scatter_internal'])
    return {'name':rows[0]['name'],'G':rows[0]['G'],'levels_used':len(valid),'p_level':p_level,'p_scatter':p_scatter,'jensen_gap_mean':float(np.mean(gaps)) if gaps else float('nan'),**{'p_'+k:v for k,v in mode_slopes.items()}}


def main(paths):
    outdir=ROOT/'epfl_results'; outdir.mkdir(exist_ok=True)
    all_rows=[]; sums=[]
    for p in paths:
        nl=parse_blif(Path(p)); print('parse',nl.name,len(nl.vertices),len(nl.nets))
        A=projection_adjacency(nl)
        levels=rcm_hierarchy(A,len(nl.vertices),min_size=8)
        rows=level_metrics(nl,levels); all_rows.extend(rows); sums.append(summarize(rows))
        print('summary',sums[-1])
    with (outdir/'levels.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0].keys()),lineterminator="\n"); w.writeheader(); w.writerows(all_rows)
    with (outdir/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sums[0].keys()),lineterminator="\n"); w.writeheader(); w.writerows(sums)
    (outdir/'summary.json').write_text(json.dumps(sums,indent=2))

if __name__=='__main__':
    main(sys.argv[1:])
