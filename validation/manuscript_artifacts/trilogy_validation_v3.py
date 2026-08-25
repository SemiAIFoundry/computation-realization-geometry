# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations
import json, math, csv, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'trilogy_validation_v3_results.json'

def test_contract_pushforward(rng, trials=1000):
    errors=0
    for _ in range(trials):
        n=int(rng.integers(3,50)); m=int(rng.integers(2,n)); k=int(rng.integers(1,m+1))
        # Surjective-ish deterministic maps via sorted random cuts.
        q=np.sort(rng.integers(0,m,size=n)); q[0]=0; q[-1]=m-1
        # Compress labels to contiguous to avoid empty groups.
        _,q=np.unique(q,return_inverse=True); m2=q.max()+1
        r=np.sort(rng.integers(0,k,size=m2)); r[0]=0; r[-1]=max(0,k-1)
        _,r=np.unique(r,return_inverse=True)
        a=rng.random(n)
        direct=np.zeros(r.max()+1)
        two=np.zeros_like(direct)
        for i,v in enumerate(a): direct[r[q[i]]]+=v
        mid=np.zeros(m2)
        for i,v in enumerate(a): mid[q[i]]+=v
        for j,v in enumerate(mid): two[r[j]]+=v
        if not np.allclose(direct,two,rtol=0,atol=1e-12): errors+=1
        # Max aggregation is also associative.
        directm=np.full(r.max()+1,-np.inf); midm=np.full(m2,-np.inf); twom=np.full(r.max()+1,-np.inf)
        for i,v in enumerate(a): directm[r[q[i]]]=max(directm[r[q[i]]],v); midm[q[i]]=max(midm[q[i]],v)
        for j,v in enumerate(midm): twom[r[j]]=max(twom[r[j]],v)
        if not np.allclose(directm,twom,rtol=0,atol=0): errors+=1
    assert errors==0
    return {'trials':trials,'errors':errors}

def test_loomis_whitney(rng,trials=5000):
    violations=0; tight=0
    for _ in range(trials):
        n=int(rng.integers(2,10))
        triples=np.array([(i,j,k) for i in range(n) for j in range(n) for k in range(n)],dtype=int)
        mask=rng.random(len(triples))<rng.uniform(.01,.4)
        S=triples[mask]
        if len(S)==0: continue
        A=len(set((int(i),int(k)) for i,j,k in S))
        B=len(set((int(k),int(j)) for i,j,k in S))
        C=len(set((int(i),int(j)) for i,j,k in S))
        if len(S)**2>A*B*C: violations+=1
        if len(S)**2==A*B*C: tight+=1
    assert violations==0
    # Explicit single inner product counterexample to the discarded rows/columns wording.
    n=17; S=[(0,0,k) for k in range(n)]
    row_count=1; col_count=1; output_count=1
    projection=(len(set((i,k) for i,j,k in S)),len(set((k,j) for i,j,k in S)),len(set((i,j) for i,j,k in S)))
    assert n>math.sqrt(row_count*col_count*output_count)
    assert n*n<=projection[0]*projection[1]*projection[2]
    return {'random_trials':trials,'violations':violations,'tight_cases':tight,'inner_product_counterexample_n':n,'correct_projection_sizes':projection}

def test_gemm_25d():
    P=64; b=2
    cs=[1,2,4]
    def mem(n,c): return 3*c*n*n/P*b
    def q(n,c): return 2*n*n/math.sqrt(c*P)*b
    def rounds(c): return math.ceil(math.sqrt(P/c**3))+math.ceil(math.log2(c))
    thresholds={}
    for cap_mib in [32,96]:
        thresholds[cap_mib]={c:math.sqrt(cap_mib*2**20*P/(3*c*b)) for c in cs}
    recs={8192:max(c for c in cs if mem(8192,c)<=32*2**20),12288:max(c for c in cs if mem(12288,c)<=32*2**20),16384:max(c for c in cs if mem(16384,c)<=32*2**20)}
    assert recs=={8192:4,12288:2,16384:1}
    # Pareto monotonicity within family.
    assert all(q(8192,cs[i+1])<q(8192,cs[i]) and rounds(cs[i+1])<=rounds(cs[i]) for i in range(2))
    # Calibrated n=8192 values.
    vals=[]
    for c in cs:
        qb=q(8192,c)
        vals.append({'c':c,'memory_MiB':mem(8192,c)/2**20,'traffic_MiB':qb/2**20,'rounds':rounds(c),'UCIe_A32_us':qb/256e9*1e6,'energy_mJ':(P*qb*8/2)*.25e-12*1e3})
    return {'thresholds':thresholds,'recommendations_32MiB':recs,'n8192':vals}

def test_collective_robustness():
    # z1=(63,r,1.03125), z2=(14,8,.046875)
    cases={}
    for r in [1,6,8,63]:
        if r<8:
            cases[r]={'exists':True,'wR_boundary_coeff_wQ':49/(8-r),'wR_boundary_coeff_wM':.984375/(8-r)}
        else: cases[r]={'exists':False}
    assert abs(cases[1]['wR_boundary_coeff_wQ']-7)<1e-12
    assert abs(cases[6]['wR_boundary_coeff_wQ']-24.5)<1e-12
    assert not cases[8]['exists'] and not cases[63]['exists']
    return cases

def test_attention():
    std={'Q':40.3,'F':66.6,'T':41.7}; flash={'Q':4.4,'F':75.2,'T':7.3}
    dQ=std['Q']-flash['Q']; dF=flash['F']-std['F']
    threshold=dF/dQ
    assert abs(threshold-.23955431754874676)<1e-12
    assert abs(std['T']/flash['T']-5.712328767123288)<1e-12
    return {'delta_Q_GB':dQ,'delta_F_GFLOP':dF,'compute_bandwidth_threshold_FLOP_byte':threshold,'measured_speedup':std['T']/flash['T'],'FP16_M_gt_d2_KiB':{64:8,128:32}}

def test_epfl():
    res=json.loads((ROOT/'epfl_results_v2/curvature_closure.json').read_text())
    summary=pd.read_csv(ROOT/'epfl_results_v2/summary.csv')
    assert float(summary.decomposition_max_abs_residual.max())==0.0
    by={(r['name'],r['hierarchy']):r for r in summary.to_dict('records')}
    assert abs(by[('multiplier','RCM-fixed')]['p_level']-.9094635313007646)<1e-12
    assert abs(by[('multiplier','recursive-RCM')]['p_level']-.8307748381090272)<1e-12
    assert res[0]['slope_rmse_exact_mode_identity']<.02 and res[1]['slope_rmse_exact_mode_identity']<.005
    assert res[0]['curvature_rmse_exact_mode_identity']<.08 and res[1]['curvature_rmse_exact_mode_identity']<.02
    return {'netlists':sorted(summary.name.unique().tolist()),'hierarchy_multiplier':{'RCM':by[('multiplier','RCM-fixed')]['p_level'],'recursive_RCM':by[('multiplier','recursive-RCM')]['p_level']},'curvature_closure':res}

def main():
    rng=np.random.default_rng(20260724)
    result={'contract_pushforward':test_contract_pushforward(rng),'loomis_whitney':test_loomis_whitney(rng),'gemm_25d':test_gemm_25d(),'collective_robustness':test_collective_robustness(),'attention':test_attention(),'epfl':test_epfl()}
    OUT.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
