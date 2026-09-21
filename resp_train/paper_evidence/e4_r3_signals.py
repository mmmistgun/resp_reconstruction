"""R3 对数幅度与 THO 的固定信号对应检查；全窗口保留未定义关联。"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from resp_train.metrics.task import TaskMetricConfig
from resp_train.protocols.respiration import canonicalize_numpy
from resp_train.paper_evidence.e4_r3_norm import TRANSFORMS


def correlation(x,y,threshold):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if x.shape!=y.shape or x.size<2: raise ValueError('关联 shape/样本数不符')
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise FloatingPointError('信号关联出现非有限值')
    x=x-x.mean(); y=y-y.mean()
    sx=np.sqrt(np.mean(x*x)); sy=np.sqrt(np.mean(y*y))
    if sx<=threshold or sy<=threshold:
        return None,'low_variance'
    return float(np.clip(np.mean(x*y)/(sx*sy),-1,1)),'defined'


def log_rms_envelopes(values,task):
    """与冻结 log-RMS 定义相同，批量滑窗避免逐尺度的 Python 窗口循环。"""
    values=np.asarray(values,dtype=np.float64)
    windows=np.lib.stride_tricks.sliding_window_view(values,task.envelope_window,axis=-1)
    windows=windows[...,::task.envelope_step,:]
    return .5*np.log(np.mean(windows*windows,axis=-1)+task.envelope_eps)


def signal_window(w,target,shifts,reference,cfg,row,plot_dir=None):
    """2 Hz 载幅图按相同 bin 中心对齐 THO；不挑尺度、不翻转极性。"""
    task=TaskMetricConfig.from_config(cfg)
    if w.shape!=(97,360) or target.size!=18000: raise ValueError('信号窗口 shape 非法')
    if not np.isfinite(w).all() or not np.isfinite(target).all(): raise FloatingPointError('信号输入非有限')
    projected,_=canonicalize_numpy(target.reshape(1,-1),fs=task.fs,low_hz=task.band_low_hz,
                                   high_hz=task.band_high_hz,scale_eps=task.scale_eps)
    tho=projected[0].reshape(360,50).mean(1)
    envelope_task=replace(task,fs=2.,length=360,envelope_window=round(task.envelope_window/task.fs*2),
                          envelope_step=round(task.envelope_step/task.fs*2))
    if envelope_task.envelope_window<1 or envelope_task.envelope_step<1: raise ValueError('包络窗口不能映射到 2 Hz')
    target_envelope=log_rms_envelopes(tho,envelope_task)
    lag_seconds=np.arange(-task.max_lag_samples,task.max_lag_samples+1)/task.fs
    records=[]; curves={}; representations=['R3_MEAN',*[f'scale_{s}' for s in range(73,97)]]
    for kind in TRANSFORMS:
        region=np.array(w[73:97],dtype=np.float64,copy=True)
        if kind=='TRAIN_MEAN': region[:]=reference[73:97,None]
        elif kind=='WINDOW_FLAT': region[:]=region.mean(1,keepdims=True)
        elif kind=='REVERSE': region=region[:,::-1].copy()
        elif kind.startswith('SHIFT_'): region=np.roll(region,int(shifts[int(kind[-1])-1]),axis=-1)
        values=np.concatenate((region.mean(0,keepdims=True),region),axis=0)
        bands,_=canonicalize_numpy(values,fs=2.,low_hz=task.band_low_hz,high_hz=task.band_high_hz,scale_eps=task.scale_eps)
        envelopes=log_rms_envelopes(bands,envelope_task)
        for j,name in enumerate(representations):
            for association,signal,reference_signal in (('cycle',bands[j],tho),('envelope',envelopes[j],target_envelope)):
                score,reason=correlation(signal,reference_signal,task.dynamic_eps)
                records.append(dict(row,transform=kind,representation=name,association=association,lag_sec=0.,
                    support='full',correlation=score,absolute_correlation=abs(score) if score is not None else None,defined=score is not None,reason=reason))
                # 2 Hz 栅格比原 ±0.3 s lag 范围粗；仅周期主曲线用明确的线性插值辅助扫描。
                # 所有 lag 使用同一中心支持区，不选择最优 lag，也不声称插值提高时间分辨率。
                if j==0 and association=='cycle':
                    step=.5
                    radius=int(np.ceil(max(abs(lag_seconds))/step))
                    if len(signal)<=2*radius+1: raise ValueError('lag profile 支持区不足')
                    start,stop=radius,len(signal)-radius
                    for lag in lag_seconds:
                        shifted=np.interp(np.arange(start,stop)+lag/step,np.arange(len(signal)),signal)
                        score,reason=correlation(shifted,reference_signal[start:stop],task.dynamic_eps)
                        records.append(dict(row,transform=kind,representation=name,association=association,lag_sec=float(lag),
                            support='common_lag_crop_linear_interpolation',correlation=score,absolute_correlation=abs(score) if score is not None else None,
                            defined=score is not None,reason=reason))
        curves[kind]=dict(raw=values[0],cycle=bands[0],envelope=envelopes[0])
    if plot_dir is not None:
        plot_curves(plot_dir,row,curves,tho,target_envelope,envelope_task)
    return records


def plot_curves(directory,row,curves,tho,target_envelope,task):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def scaled(x):
        centered=x-x.mean(); std=centered.std()
        return centered/std if std>task.dynamic_eps else np.zeros_like(centered)
    t=(np.arange(360)*50+24.5)/100
    et=.245+(np.arange(len(target_envelope))*task.envelope_step+(task.envelope_window-1)/2)/2
    fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    axes[1].plot(t,scaled(tho),color='black',lw=1.5,label='THO projected (scaled)')
    axes[2].plot(et,target_envelope,color='black',lw=1.5,label='THO log RMS')
    arrays={'time':t,'envelope_time':et,'tho':tho,'tho_envelope':target_envelope}
    for kind,data in curves.items():
        axes[0].plot(t,data['raw'],label=kind,alpha=.8)
        axes[1].plot(t,scaled(data['cycle']),label=kind,alpha=.6)
        axes[2].plot(et,data['envelope'],label=kind,alpha=.7)
        arrays.update({f'{kind}_{key}':value for key,value in data.items()})
    for ax in axes: ax.legend(ncol=4,fontsize=7); ax.set_xlabel('Time (s)')
    axes[0].set_title(f"samp {row['samp_id']} / row {row['dataset_row_id']}: R3 mean log magnitude")
    axes[1].set_ylabel('Cycle correspondence; polarity unchanged')
    axes[2].set_ylabel('Relative log-RMS envelopes')
    path=Path(directory)/f"row_{int(row['dataset_row_id'])}"
    fig.savefig(path.with_suffix('.png'),dpi=120); plt.close(fig)
    np.savez(path.with_suffix('.npz'),**arrays)


def summarize_signals(frame):
    keys=['transform','representation','association','lag_sec','support']
    records=[]
    groups=[('pooled','ALL',frame),*(('samp_id',str(s),g) for s,g in frame.groupby('samp_id'))]
    for scope,subject,part in groups:
        for key,g in part.groupby(keys,sort=False):
            records.append(dict(zip(keys,key),scope=scope,group=subject,n=len(g),defined_n=int(g.defined.sum()),
                signed_mean=g.correlation.mean(),absolute_mean=g.absolute_correlation.mean(),
                signed_median=g.correlation.median(),absolute_median=g.absolute_correlation.median()))
    return pd.DataFrame(records)


def paired_correlations(frame):
    keys=['dataset_row_id','samp_id','split','representation','association','lag_sec','support']
    baseline=frame[frame['transform'].eq('FULL')][keys+['correlation','absolute_correlation']]
    joined=frame.merge(baseline,on=keys,how='left',suffixes=('','_full'),validate='many_to_one')
    joined['signed_change']=joined.correlation-joined.correlation_full
    joined['absolute_drop']=joined.absolute_correlation_full-joined.absolute_correlation
    joined['paired_defined']=joined.correlation.notna() & joined.correlation_full.notna()
    return joined


def summarize_paired(frame):
    keys=['transform','representation','association','lag_sec','support']
    records=[]
    groups=[('pooled','ALL',frame),*(('samp_id',str(s),g) for s,g in frame.groupby('samp_id'))]
    for scope,subject,part in groups:
        for key,g in part.groupby(keys,sort=False):
            records.append(dict(zip(keys,key),scope=scope,group=subject,n=len(g),paired_defined_n=int(g.paired_defined.sum()),
                signed_change_mean=g.signed_change.mean(),absolute_drop_mean=g.absolute_drop.mean(),
                drop_positive_count=int((g.absolute_drop>0).sum()),drop_negative_count=int((g.absolute_drop<0).sum())))
    return pd.DataFrame(records)
