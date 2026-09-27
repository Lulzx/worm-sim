#!/usr/bin/env python3
"""Plot the complete capacity-test history (optional dependency: matplotlib)."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--curve',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();data=np.genfromtxt(a.curve,delimiter=',',names=True)
    plt.rcParams['svg.hashsalt']='wormsim-capacity-v1'
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    x=data['epoch'];capture=100*data['captured_start_zero_energy']
    axes[0].plot(x,capture,color='#2563eb',linewidth=1.5,label='Recorded training iterates')
    axes[0].axhline(90,color='#b45309',linestyle='--',linewidth=1,label='Predeclared capacity gate (90%)')
    best=int(np.argmin(data['mse']))
    axes[0].scatter(x[best],capture[best],color='#b45309',s=22,zorder=3,label='Best observed loss')
    axes[0].scatter(x[-1],capture[-1],color='#2563eb',s=22,zorder=3,label='Final iterate')
    axes[0].set(ylabel='Available reduction captured (%)',ylim=(0,100))
    axes[0].legend(loc='lower right',frameon=False,fontsize=9)
    axes[1].fill_between(x,data['gain_min'],data['gain_max'],color='#2563eb',alpha=.12)
    axes[1].plot(x,data['gain_max'],color='#2563eb',linewidth=1,label='Maximum gain')
    axes[1].plot(x,data['gain_min'],color='#64748b',linewidth=1,label='Minimum gain')
    axes[1].set(yscale='log',ylabel='Observation gain',xlabel='Adam updates',xlim=(0,x[-1]+10))
    axes[1].legend(loc='center left',frameon=False,fontsize=9)
    for ax in axes:
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Level 0 training capacity: ADAL / ADAR, 50 trials',fontsize=14)
    fig.text(.5,.925,'Seed 1 · learning rate 0.01 · 120-second preparation',ha='center',fontsize=10,color='#475569')
    fig.text(.5,.02,'Training only; gains are unregularized. No held-out evaluation.',ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.04,1,.92))
    metadata={'Date':None} if a.output.endswith('.svg') else None
    fig.savefig(a.output,dpi=160,metadata=metadata)

if __name__=='__main__':main()
