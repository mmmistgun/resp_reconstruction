"""从已完成汇总表生成本轮结果索引与比较图，不运行模型或重算逐窗指标。"""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import hashlib
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
SESSION=ROOT/'runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731'
VALIDATION=SESSION/'summary/attempt_20261001T154248Z_f22f23ce3ef0'
TEST=SESSION/'research_test/summary/attempt_20261002T054624Z_e22f113cfb30'
MECHANISMS=SESSION/'research_test/mechanisms_r2/revision_485f7f748f79/summary/attempt_20261002T084725Z_430651990a29'


def main():
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
    sources={}
    def read(path):
        raw=path.read_bytes()
        sources[str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
        return pd.read_csv(path)
    output=ROOT/'runs/cwt_apor_v2/reports'/('summary_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    tables={}
    for view,path in [('validation',VALIDATION),*[(v,TEST/v) for v in ['full','exclude670','subject670']]]:
        frame=read(path/'across_seed.csv')
        means=frame.pivot(index='arm',columns='metric',values='seed_mean')
        tables[view]=means
        frame.assign(view=view).to_csv(output/f'{view}_metrics.csv',index=False)
    combined=pd.concat({view:table for view,table in tables.items()},axis=1)
    combined.columns=[f'{view}__{metric}' for view,metric in combined.columns]
    combined.to_csv(output/'all_configurations.csv')
    for view in ['full','exclude670','subject670']:
        read(MECHANISMS/view/'across_seed.csv').to_csv(output/f'mechanisms_{view}.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    labels={'A0':'A0（0.5秒 / 8 Hz）','H':'H-only','P_025':'池化0.25秒','P_100':'池化1秒',
            'C_20':'上限20 Hz','C_4':'上限4 Hz','Q_mu20_v24':'μ20 / v24','Q_mu6_v24':'μ6 / v24'}
    offsets={'A0':(7,-16),'H':(7,-16),'P_025':(7,-14),'P_100':(5,9),'C_20':(7,6),
             'C_4':(7,6),'Q_mu20_v24':(-72,-12),'Q_mu6_v24':(6,9)}
    groups=[('Q_', 'CWT参数', '#7a8794','o'),('P_','时间池化','#3c8d70','s'),
            ('C_','频率上限','#447ab4','^'),('H','仅高频条件','#b47a29','D')]
    for ax,view,title in zip(axes,['full','exclude670'],['完整test：2310窗 / 8受试者','排除670：2231窗 / 7受试者']):
        table=tables[view]
        for prefix,label,color,marker in groups:
            part=table.loc[table.index.str.startswith(prefix)]
            ax.scatter(part.local_rr_mae_bpm,part.envelope_trajectory_mae,c=color,marker=marker,s=43,label=label)
        baseline=table.loc['A0']
        ax.scatter([baseline.local_rr_mae_bpm],[baseline.envelope_trajectory_mae],c='#b94246',marker='*',s=160,label='A0基准',zorder=4)
        ax.axvline(baseline.local_rr_mae_bpm,color='#b94246',lw=.7,alpha=.35)
        ax.axhline(baseline.envelope_trajectory_mae,color='#b94246',lw=.7,alpha=.35)
        for arm,label in labels.items():
            row=table.loc[arm]
            ax.annotate(label,(row.local_rr_mae_bpm,row.envelope_trajectory_mae),xytext=offsets[arm],
                        textcoords='offset points',fontsize=7)
        ax.set(title=title,xlabel='Local RR MAE（bpm，越低越好）',ylabel='包络轨迹 MAE（越低越好）')
        ax.grid(alpha=.15)
        ax.margins(x=.18,y=.20)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncol=5,frameon=False,fontsize=8)
    fig.suptitle('APOR/A0：节律与包络重建的取舍（三seed均值）',fontsize=12)
    fig.savefig(output/'rhythm_envelope_tradeoff.png',dpi=180)
    fig.savefig(output/'rhythm_envelope_tradeoff.pdf')
    plt.close(fig)
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (output/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    main()
