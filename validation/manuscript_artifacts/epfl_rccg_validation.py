# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
import csv, json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def _resolve_netlist_dir() -> Path:
    configured = os.environ.get("CRG_EPFL_NETLIST_DIR")
    candidates = [
        Path(configured).expanduser().resolve() if configured else None,
        ROOT.parents[1] / "third_party" / "epfl" / "netlists",
        ROOT / "third_party_epfl_netlists",  # temporary reproduction workspace
        ROOT / "source_netlists",            # historical-layout fallback
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "EPFL netlists not found; set CRG_EPFL_NETLIST_DIR or restore third_party/epfl/netlists"
    )

NETLIST_DIR=_resolve_netlist_dir()
sys.path.insert(0,str(ROOT))
from epfl_spectrum_analysis import (
    parse_blif, projection_adjacency, rcm_hierarchy, recursive_rcm_hierarchy,
    fanout_class, ols_slope
)

CLASSES=['2-pin','3-8 pin','9-32 pin','>32 pin']


def level_metrics_internal(nl, levels, hierarchy:str):
    rec=[]
    for depth,mods in enumerate(levels):
        assign=np.empty(len(nl.vertices),dtype=int)
        sizes=[]
        for j,m in enumerate(mods):
            assign[m]=j; sizes.append(len(m))
        k=len(mods)
        per_total=np.zeros(k,dtype=int)
        per_internal=np.zeros(k,dtype=int)
        per_io=np.zeros(k,dtype=int)
        class_internal={c:np.zeros(k,dtype=int) for c in CLASSES}
        class_pad={c:np.zeros(k,dtype=int) for c in CLASSES}
        for name,pins in nl.nets.items():
            touched=np.unique(assign[np.asarray(pins,dtype=int)])
            q=len(touched)
            cls=fanout_class(len(pins))
            is_io=name in nl.io_nets
            if q>=2:
                per_total[touched]+=1
                class_pad[cls][touched]+=1
                if is_io:
                    per_io[touched]+=1
                else:
                    per_internal[touched]+=1
                    class_internal[cls][touched]+=1
            if is_io and q==1:  # pad-inclusive boundary convention
                per_total[touched]+=1
                per_io[touched]+=1
                class_pad[cls][touched]+=1
        G=len(nl.vertices); m=len(mods); g=G/m
        row={
            'name':nl.name,'hierarchy':hierarchy,'depth':depth,'modules':m,'G':G,
            'g_nominal':g,'size_min':min(sizes),'size_max':max(sizes),
            'size_mean':float(np.mean(sizes)),
            'T_level_mean':float(np.mean(per_total)),
            'T_level_mean_internal':float(np.mean(per_internal)),
            'T_io_mean':float(np.mean(per_io)),
            'logT_scatter_internal':float(np.mean(np.log(np.maximum(per_internal,1)))),
        }
        for c in CLASSES:
            row['T_internal_'+c]=float(np.mean(class_internal[c]))
            row['T_pad_'+c]=float(np.mean(class_pad[c]))
        # exact decomposition audit
        row['mode_sum_internal']=sum(row['T_internal_'+c] for c in CLASSES)
        row['decomposition_residual']=row['T_level_mean_internal']-row['mode_sum_internal']
        rec.append(row)
    return rec


def summary(rows):
    valid=[r for r in rows if 16<=r['g_nominal']<=r['G']/4 and r['T_level_mean_internal']>0]
    x=np.log([r['g_nominal'] for r in valid])
    y=np.log([r['T_level_mean_internal'] for r in valid])
    ys=np.array([r['logT_scatter_internal'] for r in valid])
    out={
        'name':rows[0]['name'],'hierarchy':rows[0]['hierarchy'],'G':rows[0]['G'],
        'levels_used':len(valid),
        'p_level':ols_slope(x,y) if len(x)>=2 else float('nan'),
        'p_scatter':ols_slope(x,ys) if len(x)>=2 else float('nan'),
        'jensen_gap_mean':float(np.mean(y-ys)) if len(x) else float('nan'),
        'decomposition_max_abs_residual':max(abs(r['decomposition_residual']) for r in rows),
    }
    for c in CLASSES:
        vv=[r for r in valid if r['T_internal_'+c]>0]
        out['p_'+c]=ols_slope(
            np.log([r['g_nominal'] for r in vv]),
            np.log([r['T_internal_'+c] for r in vv])) if len(vv)>=2 else float('nan')
    return out


def curvature_closure(rows, out_prefix:Path):
    d=pd.DataFrame(rows)
    d=d[(d.g_nominal>=16)&(d.g_nominal<=d.G.iloc[0]/4)&(d.T_level_mean_internal>0)].copy()
    d=d.sort_values('g_nominal')
    x=np.log(d.g_nominal.to_numpy(float))
    total=d.T_level_mean_internal.to_numpy(float)
    modes=np.stack([d['T_internal_'+c].to_numpy(float) for c in CLASSES],axis=1)
    # Keep modes present at >=3 levels, with a tiny numerical floor only for fitting.
    keep=[j for j in range(len(CLASSES)) if np.count_nonzero(modes[:,j]>0)>=3]
    classes=[CLASSES[j] for j in keep]; modes=modes[:,keep]
    # Fit one power law per measured communication mode over the declared window.
    p=[]; a=[]
    for j in range(modes.shape[1]):
        mask=modes[:,j]>0
        pj,logaj=np.polyfit(x[mask],np.log(modes[mask,j]),1)
        p.append(float(pj)); a.append(float(np.exp(logaj)))
    p=np.asarray(p); a=np.asarray(a)
    pred_modes=np.exp(np.outer(x,p))*a
    pred_total=pred_modes.sum(axis=1)
    pred_weights=pred_modes/pred_total[:,None]
    pred_slope=pred_weights@p
    pred_curvature=((p[None,:]-pred_slope[:,None])**2*pred_weights).sum(axis=1)
    # Observed local slope and curvature from the measured total profile.
    obs_log=np.log(total)
    obs_slope=np.gradient(obs_log,x,edge_order=2)
    obs_curv=np.gradient(obs_slope,x,edge_order=2)
    # Also exact algebraic slope using measured mode interval slopes, to audit accounting.
    log_modes=np.log(np.maximum(modes,1e-15))
    local_mode_slope=np.gradient(log_modes,x,axis=0,edge_order=2)
    measured_weights=modes/np.maximum(total[:,None],1e-15)
    exact_mix_slope=(measured_weights*local_mode_slope).sum(axis=1)
    mode_drift=np.gradient(local_mode_slope,x,axis=0,edge_order=2)
    exact_mix_curv=(measured_weights*(local_mode_slope-exact_mix_slope[:,None])**2).sum(axis=1)+(measured_weights*mode_drift).sum(axis=1)
    result={
        'name':rows[0]['name'],'hierarchy':rows[0]['hierarchy'],'classes':classes,
        'mode_exponents':{c:float(v) for c,v in zip(classes,p)},
        'levels':len(x),
        'profile_relative_rmse':float(np.sqrt(np.mean(((pred_total-total)/total)**2))),
        'slope_rmse_power_mode':float(np.sqrt(np.mean((pred_slope-obs_slope)**2))),
        'curvature_rmse_power_mode':float(np.sqrt(np.mean((pred_curvature-obs_curv)**2))),
        'slope_rmse_exact_mode_identity':float(np.sqrt(np.mean((exact_mix_slope-obs_slope)**2))),
        'curvature_rmse_exact_mode_identity':float(np.sqrt(np.mean((exact_mix_curv-obs_curv)**2))),
        'max_decomposition_residual':float(np.max(np.abs(total-modes.sum(axis=1)))),
    }
    out=pd.DataFrame({
        'g':np.exp(x),'observed_total':total,'predicted_total_power_modes':pred_total,
        'observed_local_slope':obs_slope,'predicted_slope_power_modes':pred_slope,
        'exact_mixture_slope_measured_modes':exact_mix_slope,
        'observed_curvature':obs_curv,'predicted_curvature_power_modes':pred_curvature,
        'exact_curvature_measured_modes':exact_mix_curv,
    })
    out.to_csv(out_prefix.with_suffix('.csv'),index=False)
    fig,axs=plt.subplots(3,1,figsize=(6.6,8.3),sharex=True)
    axs[0].loglog(np.exp(x),total,'o-',label='measured internal total')
    axs[0].loglog(np.exp(x),pred_total,'--',label='sum of fitted mode laws')
    axs[0].set_ylabel('mean terminals'); axs[0].legend(fontsize=8); axs[0].grid(True,which='both',alpha=.25)
    axs[1].semilogx(np.exp(x),obs_slope,'o-',label='measured local slope')
    axs[1].semilogx(np.exp(x),pred_slope,'--',label='predicted from fitted mode exponents')
    axs[1].semilogx(np.exp(x),exact_mix_slope,':',label='measured-mode identity')
    axs[1].set_ylabel('$p_{loc}$'); axs[1].legend(fontsize=7); axs[1].grid(True,which='both',alpha=.25)
    axs[2].semilogx(np.exp(x),obs_curv,'o-',label='measured curvature')
    axs[2].semilogx(np.exp(x),pred_curvature,'--',label='variance of fitted mode exponents')
    axs[2].semilogx(np.exp(x),exact_mix_curv,':',label='variance + measured mode drift')
    axs[2].set_ylabel(r'$d p_{loc}/d\log g$'); axs[2].set_xlabel('nominal module size $g$')
    axs[2].legend(fontsize=7); axs[2].grid(True,which='both',alpha=.25)
    fig.suptitle(f"EPFL {rows[0]['name']}: closed-loop mode attribution ({rows[0]['hierarchy']})")
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(out_prefix.with_suffix('.pdf')); fig.savefig(out_prefix.with_suffix('.png'),dpi=180)
    plt.close(fig)
    return result


def main():
    files=['adder','bar','arbiter','multiplier','router','i2c','div','log2','sqrt','square']
    outdir=ROOT/'epfl_results_v2'; outdir.mkdir(exist_ok=True)
    figdir=ROOT/'figures_epfl_v2'; figdir.mkdir(exist_ok=True)
    for stale in figdir.glob('multiplier_curvature_spectral-cut.*'):
        stale.unlink()
    all_rows=[]; summaries=[]; bykey={}
    for name in files:
        nl=parse_blif(NETLIST_DIR/(name+'.blif'))
        A=projection_adjacency(nl)
        for hierarchy,levels in [('RCM-fixed',rcm_hierarchy(A,len(nl.vertices),min_size=8))]:
            rows=level_metrics_internal(nl,levels,hierarchy); all_rows.extend(rows); summaries.append(summary(rows)); bykey[(name,hierarchy)]=rows
        if name in {'bar','multiplier'}:
            levels=recursive_rcm_hierarchy(A,len(nl.vertices),min_size=8)
            rows=level_metrics_internal(nl,levels,'recursive-RCM'); all_rows.extend(rows); summaries.append(summary(rows)); bykey[(name,'recursive-RCM')]=rows
        print(name,len(nl.vertices),summaries[-1]); import gc; del A; gc.collect()
    pd.DataFrame(all_rows).to_csv(outdir/'levels.csv',index=False)
    pd.DataFrame(summaries).to_csv(outdir/'summary.csv',index=False)
    (outdir/'summary.json').write_text(json.dumps(summaries,indent=2))
    closures=[]
    for hierarchy in ['RCM-fixed','recursive-RCM']:
        res=curvature_closure(bykey[('multiplier',hierarchy)],figdir/f'multiplier_curvature_{hierarchy.lower()}')
        closures.append(res)
    (outdir/'curvature_closure.json').write_text(json.dumps(closures,indent=2))
    # hierarchy comparison figure
    s=pd.DataFrame(summaries)
    sel=s[s.name.isin(['bar','multiplier'])].copy()
    fig,ax=plt.subplots(figsize=(6.2,3.7))
    names=['bar','multiplier']; xpos=np.arange(len(names)); width=.34
    for i,h in enumerate(['RCM-fixed','recursive-RCM']):
        vals=[float(sel[(sel.name==n)&(sel.hierarchy==h)].p_level.iloc[0]) for n in names]
        ax.bar(xpos+(i-.5)*width,vals,width,label=h)
    ax.set_xticks(xpos, names); ax.set_ylabel('fitted level-mean exponent'); ax.set_ylim(0,1.05)
    ax.set_title('Hierarchy choice changes the measured Rent summary')
    ax.legend(fontsize=8); ax.grid(axis='y',alpha=.25); fig.tight_layout()
    fig.savefig(figdir/'hierarchy_comparison.pdf'); fig.savefig(figdir/'hierarchy_comparison.png',dpi=180); plt.close(fig)
    print(json.dumps(closures,indent=2))

if __name__=='__main__': main()
