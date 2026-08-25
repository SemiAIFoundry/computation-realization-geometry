# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures_flagship_v2'; FIG.mkdir(exist_ok=True)
OUT=ROOT/'flagship_results_v2'; OUT.mkdir(exist_ok=True)

# Public technology anchors used exactly as declared in the manuscripts.
TECH={
 'UCIe-S-32': {'bw_GBs':64.0,'energy_pJ_bit':0.50,'source':'UCIe 2.0 target, standard package, one width-16 cluster'},
 'UCIe-A-32': {'bw_GBs':256.0,'energy_pJ_bit':0.25,'source':'UCIe 2.0 target, advanced package, one width-64 cluster'},
 'UCIe-A-64': {'bw_GBs':512.0,'energy_pJ_bit':0.25,'source':'UCIe 3.0 rate; energy held at UCIe 2.0 public target as a declared proxy'},
}
P=64
WORD_BYTES=2  # BF16/FP16
C_VALUES=[1,2,4]  # legal 2.5D replication factors for P=64: c <= P^(1/3)
SRAM_MIB=[32,96]  # public AMD CCD L3 anchors


def gemm_memory_bytes(n,c):
    return 3*c*n*n/P*WORD_BYTES

def gemm_traffic_bytes(n,c):
    # Explicit constructive leading model: two matrix streams along critical path.
    return 2*n*n/math.sqrt(c*P)*WORD_BYTES

def gemm_rounds(c):
    # Solomonik-Demmel asymptotic stage model, made discrete for P=64.
    return int(math.ceil(math.sqrt(P/(c**3))) + math.ceil(math.log2(c)))

def gemm_rows():
    rows=[]
    for n in [4096,8192,12288,16384,18432]:
        for c in C_VALUES:
            mem=gemm_memory_bytes(n,c)
            q=gemm_traffic_bytes(n,c)
            row={'n':n,'P':P,'c':c,'memory_MiB':mem/2**20,'traffic_MiB_per_chiplet':q/2**20,'rounds_model':gemm_rounds(c)}
            for name,t in TECH.items():
                row[f'time_us_{name}']=q/(t['bw_GBs']*1e9)*1e6
                # Package-wide transported-bit energy; divide by two to avoid sender/receiver double-counting.
                row[f'energy_mJ_{name}']=(P*q*8/2)*t['energy_pJ_bit']*1e-12*1e3
            rows.append(row)
    return pd.DataFrame(rows)

def phase_map():
    # Largest feasible c is Pareto optimal in this 2.5D family because q and r decrease with c.
    ns=np.arange(2048,24577,128)
    map_rows=[]
    for sram in SRAM_MIB:
        cap=sram*2**20
        for n in ns:
            feasible=[c for c in C_VALUES if gemm_memory_bytes(n,c)<=cap]
            best=max(feasible) if feasible else 0
            map_rows.append({'n':int(n),'SRAM_MiB':sram,'c_star':best})
    return pd.DataFrame(map_rows)

def plot_gemm(df,pm):
    # Phase map
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    for sram,grp in pm.groupby('SRAM_MiB'):
        ax.step(grp.n,grp.c_star,where='post',label=f'{sram} MiB/chiplet')
    ax.set_yticks([0,1,2,4],['infeasible','1 (2D)','2 (2.5D)','4 (3D endpoint)'])
    ax.set_xlabel('matrix dimension $n$ (BF16/FP16)'); ax.set_ylabel('largest feasible replication $c^*$')
    ax.set_title('Certified 2.5D GEMM memory-replication phase map, $P=64$')
    ax.grid(True,alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG/'gemm_25d_memory_phase.pdf'); fig.savefig(FIG/'gemm_25d_memory_phase.png',dpi=180); plt.close(fig)
    # Technology-calibrated time/energy at n=8192, 32MiB
    d=df[df.n==8192].copy()
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.7))
    x=np.arange(len(C_VALUES)); width=.24
    for i,name in enumerate(TECH):
        vals=[float(d[d.c==c][f'time_us_{name}'].iloc[0]) for c in C_VALUES]
        axs[0].bar(x+(i-1)*width,vals,width,label=name)
        vals2=[float(d[d.c==c][f'energy_mJ_{name}'].iloc[0]) for c in C_VALUES]
        axs[1].bar(x+(i-1)*width,vals2,width,label=name)
    for ax in axs:
        ax.set_xticks(x,[f'c={c}' for c in C_VALUES]); ax.grid(axis='y',alpha=.25)
    axs[0].set_ylabel('ideal one-cluster link time per chiplet (µs)')
    axs[1].set_ylabel('package transported-bit energy (mJ)')
    axs[0].set_title('Bandwidth projection'); axs[1].set_title('Energy projection')
    axs[0].legend(fontsize=7)
    fig.suptitle('GEMM $n=8192$, $P=64$: calibrated UCIe projections')
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(FIG/'gemm_ucie_calibration.pdf'); fig.savefig(FIG/'gemm_ucie_calibration.png',dpi=180); plt.close(fig)

# Attention flagship: published A100 forward+backward measurement in FlashAttention Fig. 2.
ATTN={
 'standard': {'GFLOP':66.6,'HBM_GB':40.3,'runtime_ms':41.7},
 'flash': {'GFLOP':75.2,'HBM_GB':4.4,'runtime_ms':7.3},
}

def attention_analysis():
    dF=ATTN['flash']['GFLOP']-ATTN['standard']['GFLOP']
    dQ=ATTN['standard']['HBM_GB']-ATTN['flash']['HBM_GB']
    threshold=dF/dQ # effective compute-throughput / memory-bandwidth, FLOP/byte
    speedup=ATTN['standard']['runtime_ms']/ATTN['flash']['runtime_ms']
    # Phase plane: x = compute price per GFLOP / IO price per GB; y = storage price proxy.
    x=np.linspace(0,8,321)
    s=np.linspace(0,30,241) # normalized premium on materialized N^2 state
    X,S=np.meshgrid(x,s)
    # Flash wins if 35.9 IO savings + storage saving > 8.6 compute premium.
    win=(dQ + S > dF*X)
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    ax.contourf(X,S,win.astype(int),levels=[-.5,.5,1.5],alpha=.35)
    ax.plot(x,np.maximum(0,dF*x-dQ),label='break-even')
    ax.text(6.5,4,'standard materialization',ha='center')
    ax.text(1.3,22,'FlashAttention\nrecompute + tile',ha='center')
    ax.set_xlabel('relative compute price $w_F/w_Q$ (GB/GFLOP units)')
    ax.set_ylabel('normalized price of materialized $N^2$ state')
    ax.set_title('Attention RCCG phase map from published A100 signatures')
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(FIG/'attention_phase_map.pdf'); fig.savefig(FIG/'attention_phase_map.png',dpi=180); plt.close(fig)
    # IO scaling crossover M=d^2 and storage scaling.
    ds=[64,128]; Ns=np.logspace(2,5,200)
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.7))
    for d in ds:
        # normalized accesses with M=100KB/2B elements
        M=100*1024/2
        q_std=Ns*d+Ns**2
        q_flash=Ns**2*d**2/M
        axs[0].loglog(Ns,q_std,label=f'standard, d={d}')
        axs[0].loglog(Ns,q_flash,'--',label=f'flash, d={d}')
        axs[1].loglog(Ns,Ns**2,label=f'standard $N^2$, d={d}')
        axs[1].loglog(Ns,Ns,label=f'flash $N$, d={d}')
    axs[0].set_title('HBM-access scaling at 100 KiB SRAM'); axs[0].set_ylabel('normalized HBM accesses')
    axs[1].set_title('Additional-state scaling'); axs[1].set_ylabel('normalized extra state')
    for ax in axs:
        ax.set_xlabel('sequence length $N$'); ax.grid(True,which='both',alpha=.25); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG/'attention_io_storage_scaling.pdf'); fig.savefig(FIG/'attention_io_storage_scaling.png',dpi=180); plt.close(fig)
    return {'published_signatures':ATTN,'delta_GFLOP':dF,'delta_HBM_GB':dQ,'compute_to_bandwidth_break_even_FLOP_per_byte':threshold,'measured_speedup':speedup,'SRAM_crossover_elements_d_squared':{str(d):d*d for d in ds},'SRAM_crossover_KiB_FP16':{str(d):d*d*2/1024 for d in ds}}

def main():
    df=gemm_rows(); pm=phase_map(); plot_gemm(df,pm)
    df.to_csv(OUT/'gemm_25d_technology.csv',index=False,lineterminator="\n"); pm.to_csv(OUT/'gemm_memory_phase.csv',index=False,lineterminator="\n")
    att=attention_analysis()
    thresholds=[]
    for sram in SRAM_MIB:
        for c in C_VALUES:
            nmax=math.sqrt(sram*2**20*P/(3*c*WORD_BYTES))
            thresholds.append({'SRAM_MiB':sram,'c':c,'n_max':nmax})
    summary={
      'technology_library':TECH,
      'gemm':{'P':P,'word_bytes':WORD_BYTES,'c_values':C_VALUES,'memory_thresholds':thresholds,
              'recommendation_n8192_32MiB':'c=4 is the largest feasible and Pareto-dominant member of the declared 2.5D family'},
      'attention':att,
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
