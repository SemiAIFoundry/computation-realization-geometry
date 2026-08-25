#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Cross-hierarchy predictive transfer test for EPFL multiplier fanout modes.

Fits one power law per fanout mode on one hierarchy, transfers the fitted slopes
and relative mode structure to the other hierarchy, and permits only one total
amplitude calibration at the largest common scale.  This is a falsifiable
prediction of cross-hierarchy shape, not the algebraic identity R=sum_j R_j.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ['2-pin','3-8 pin','9-32 pin','>32 pin']


def prep(d: pd.DataFrame, hierarchy: str) -> pd.DataFrame:
    x=d[(d.name=='multiplier')&(d.hierarchy==hierarchy)].copy()
    x=x[(x.g_nominal>=16)&(x.g_nominal<=x.G.iloc[0]/4)&(x.T_level_mean_internal>0)]
    return x.sort_values('g_nominal')


def fit_modes(src: pd.DataFrame):
    x=np.log(src.g_nominal.to_numpy(float)); params={}
    for c in CLASSES:
        y=src['T_internal_'+c].to_numpy(float); mask=y>0
        if mask.sum()>=3:
            p,loga=np.polyfit(x[mask],np.log(y[mask]),1)
            params[c]={'amplitude':float(np.exp(loga)),'exponent':float(p)}
    return params


def transfer(params, target: pd.DataFrame):
    g=target.g_nominal.to_numpy(float)
    pred=np.zeros(len(g))
    for c,v in params.items(): pred += v['amplitude']*g**v['exponent']
    obs=target.T_level_mean_internal.to_numpy(float)
    scale=float(obs[-1]/pred[-1])  # one declared amplitude anchor at largest common scale
    pred*=scale
    lx=np.log(g)
    obs_s=np.gradient(np.log(obs),lx,edge_order=2)
    pred_s=np.gradient(np.log(pred),lx,edge_order=2)
    return {
        'one_anchor_scale_factor': scale,
        'profile_relative_rmse': float(np.sqrt(np.mean(((pred-obs)/obs)**2))),
        'slope_rmse': float(np.sqrt(np.mean((pred_s-obs_s)**2))),
        'g': g.tolist(), 'observed_total': obs.tolist(), 'predicted_total': pred.tolist(),
        'observed_slope': obs_s.tolist(), 'predicted_slope': pred_s.tolist(),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('levels_csv', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ns=ap.parse_args()
    d=pd.read_csv(ns.levels_csv)
    out={}
    for src_h,tgt_h in [('RCM-fixed','recursive-RCM'),('recursive-RCM','RCM-fixed')]:
        params=fit_modes(prep(d,src_h))
        out[f'{src_h}_to_{tgt_h}']={'fit_mode_parameters':params, **transfer(params,prep(d,tgt_h))}
    ns.out.parent.mkdir(parents=True,exist_ok=True)
    ns.out.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:{'profile_relative_rmse':v['profile_relative_rmse'],'slope_rmse':v['slope_rmse'],'one_anchor_scale_factor':v['one_anchor_scale_factor']} for k,v in out.items()},indent=2))

if __name__=='__main__': main()
