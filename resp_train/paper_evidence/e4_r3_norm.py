"""W0 R3 时序干预与逐窗口 GN 统计重放；只允许 eval/no_grad。"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TRANSFORMS = ('FULL','TRAIN_MEAN','WINDOW_FLAT','SHIFT_1','SHIFT_2','SHIFT_3','REVERSE')
MODES = ('NAT','GN1_FIXED','ALL_W_GN_FIXED')
CONDITIONS = tuple(f'{t}__{m}' for t in TRANSFORMS for m in MODES) + ('STAT_ONLY_TRAIN_MEAN',)
GN_NAMES = ('norm','temporal.blocks.0.norm','temporal.blocks.1.norm','temporal.blocks.2.norm')
R3 = slice(73,97)


def finite(x):
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError('R3 诊断出现非有限 tensor')


def make_shifts(count):
    generator=np.random.Generator(np.random.PCG64(20260920))
    return np.stack([generator.choice(np.arange(60,301),3,replace=False) for _ in range(count)])


def transform_w(w,kind,reference,shifts):
    if kind not in TRANSFORMS or w.ndim!=3 or tuple(w.shape[1:])!=(97,360):
        raise ValueError('R3 变换/shape 非法')
    finite(w)
    if kind=='FULL':
        return w
    result=w.clone(); region=w[:,R3]
    if kind=='TRAIN_MEAN':
        if reference.shape!=(97,): raise ValueError('训练参考 shape 非法')
        finite(reference)
        result[:,R3]=reference[R3].to(w)[None,:,None]
    elif kind=='WINDOW_FLAT':
        result[:,R3]=region.mean(-1,keepdim=True)
    elif kind=='REVERSE':
        result[:,R3]=region.flip(-1)
    else:
        if tuple(shifts.shape)!=(len(w),3) or shifts.dtype!=torch.int64 or bool(((shifts<60)|(shifts>300)).any()):
            raise ValueError('错位数组必须为 B×3 的冻结整数偏移')
        shift=shifts[:,int(kind[-1])-1].to(w.device)
        index=(torch.arange(360,device=w.device)[None,:]-shift[:,None])%360
        result[:,R3]=region.gather(2,index[:,None,:].expand(-1,24,-1))
    finite(result)
    return result


@dataclass
class GNStats:
    mean: torch.Tensor
    rstd: torch.Tensor

    def detached(self):
        return GNStats(self.mean.detach(),self.rstd.detach())


def native_stats(module,x,y):
    """用原生输出 dtype 重放 kernel，兼容 autocast 将 GN 提升到 FP32。"""
    with torch.autocast(x.device.type,enabled=False):
        value=x.to(y.dtype).contiguous()
        _,mean,rstd=torch.native_group_norm(value,module.weight,module.bias,len(x),x.shape[1],
            x[0,0].numel(),module.num_groups,module.eps)
    finite(mean); finite(rstd)
    if bool((rstd<=0).any()): raise FloatingPointError('GN 逆标准差非正')
    return GNStats(mean,rstd).detached()


def frozen_norm(module,x,stats,dtype):
    n,c=x.shape[:2]; groups=module.num_groups
    if stats.mean.shape!=(n,groups) or stats.rstd.shape!=(n,groups):
        raise ValueError('GN 统计的窗口/组 shape 不符')
    finite(stats.mean); finite(stats.rstd)
    if bool((stats.rstd<=0).any()): raise FloatingPointError('GN 逆标准差非正')
    # 原生 GN 等价于逐样本/通道 affine。addcmul 保留融合乘加，避免多步舍入。
    with torch.autocast(x.device.type,enabled=False):
        mean=stats.mean.float().repeat_interleave(c//groups,1)
        rstd=stats.rstd.float().repeat_interleave(c//groups,1)
        scale=rstd*module.weight.float()[None,:]
        # CUDA 原生 GN 的 fused-params kernel 使用 -scale*mean+beta；CPU 保留已验收的算术路径。
        bias=(torch.addcmul(module.bias.float()[None,:],-scale,mean) if x.is_cuda
              else module.bias.float()[None,:]-mean*scale)
        shape=(n,c,*([1]*(x.ndim-2)))
        output=torch.addcmul(bias.reshape(shape),x.float(),scale.reshape(shape)).to(dtype)
    finite(output)
    return output


@dataclass
class Trace:
    natural: dict[str,GNStats]=field(default_factory=dict)
    used: dict[str,GNStats]=field(default_factory=dict)
    points: dict[str,torch.Tensor]=field(default_factory=dict)
    replay_max_abs: dict[str,float]=field(default_factory=dict)


class NormIntervention(nn.Module):
    def __init__(self,model,condition,reference,shifts,baseline=None,mean_trace=None):
        super().__init__()
        if condition not in CONDITIONS: raise ValueError('未知 R3 条件')
        self.model,self.condition=model,condition
        self.reference,self.shifts=reference,shifts
        self.baseline,self.mean_trace=baseline,mean_trace
        self.trace=Trace()

    def forward(self,x,*,tf,**kwargs):
        if self.training or self.model.training or torch.is_grad_enabled():
            raise RuntimeError('R3 诊断仅允许 eval/no_grad')
        branch=self.model.branches['w']
        actual={name:module for name,module in branch.named_modules() if isinstance(module,nn.GroupNorm)}
        if tuple(actual)!=GN_NAMES: raise ValueError('W 分支 GN 拓扑漂移')
        if self.condition=='STAT_ONLY_TRAIN_MEAN':
            kind,mode='FULL','STAT_ONLY'
        else:
            kind,mode=self.condition.split('__')
        if mode in ('GN1_FIXED','ALL_W_GN_FIXED') and self.baseline is None:
            raise ValueError('缺少同窗口 FULL 统计')
        if mode=='STAT_ONLY' and self.mean_trace is None:
            raise ValueError('缺少同窗口 TRAIN_MEAN 自然统计')
        trace=self.trace=Trace(); handles=[]
        def gn_hook(name,module):
            def hook(_,args,output):
                stats=native_stats(module,args[0],output)
                trace.natural[name]=stats
                selected=stats
                if mode=='ALL_W_GN_FIXED' or (name=='norm' and mode=='GN1_FIXED'):
                    selected=self.baseline.natural[name]
                elif name=='norm' and mode=='STAT_ONLY':
                    selected=self.mean_trace.natural[name]
                trace.used[name]=selected
                # 同源重放也验收，不用“统计相同时返回原输出”的捷径掩盖差异。
                same=frozen_norm(module,args[0],stats,output.dtype)
                try:
                    torch.testing.assert_close(same,output,rtol=1e-5,atol=1e-6)
                except AssertionError as exc:
                    exc.add_note(f'GN 同源重放失败: layer={name}, condition={self.condition}')
                    raise
                trace.replay_max_abs[name]=float((same.float()-output.float()).abs().max())
                result=output if selected is stats else frozen_norm(module,args[0],selected,output.dtype)
                if name=='norm': trace.points['gn_front']=result.detach()
                return result
            return hook
        for name,module in actual.items():
            handles.append(module.register_forward_hook(gn_hook(name,module)))
        handles.append(branch.conv_in.register_forward_hook(lambda _,args,out:trace.points.update(conv_in=out.detach())))
        handles.append(branch.conv_out.register_forward_hook(lambda _,args,out:trace.points.update(z=F.silu(out).mean(2).detach())))
        handles.append(branch.register_forward_hook(lambda _,args,out:trace.points.update(gamma=out[0].detach(),beta=out[1].detach())))
        try:
            changed=transform_w(tf['w'],kind,self.reference,self.shifts)
            result=self.model(x,tf=dict(tf,w=changed),**kwargs)
        except BaseException as exc:
            exc.add_note(f'R3 condition={self.condition}, batch={len(x)}')
            raise
        finally:
            for handle in handles: handle.remove()
        for value in result.values(): finite(value)
        if tuple(trace.natural)!=GN_NAMES: raise RuntimeError('GN 捕获未完整覆盖 forward')
        return result


def rms_changes(value,baseline):
    """按小块规约，避免留存多个完整 FP32 前端特征副本。"""
    records=[]
    for start in range(0,len(value),8):
        x=value[start:start+8].float().flatten(1)
        b=baseline[start:start+8].float().flatten(1)
        rms=b.square().mean(1).sqrt(); delta=(x-b).square().mean(1).sqrt()
        records.extend(zip(rms.cpu().tolist(),delta.cpu().tolist()))
    return records


def describe(trace,baseline,rows,condition,epsilons):
    records=[]; stats_records=[]
    for i,row in enumerate(rows):
        records.append(dict(row,condition=condition))
    for name,x in trace.points.items():
        finite(x)
        for rec,(base,delta) in zip(records,rms_changes(x,baseline.points[name])):
            rec[f'{name}_baseline_rms']=base; rec[f'{name}_delta_rms']=delta
            rec[f'{name}_relative_delta']=delta/base if base else None
        if name in ('conv_in','gn_front'):
            for rec,(_,delta) in zip(records,rms_changes(x[:,:,:71],baseline.points[name][:,:,:71])):
                rec[f'{name}_protected_delta_rms']=delta
    for name,stats in trace.natural.items():
        origin=baseline.natural[name]; used=trace.used[name]
        arrays=[v.float().cpu().numpy() for v in (stats.mean,stats.rstd,origin.mean,origin.rstd,used.mean,used.rstd)]
        for i,row in enumerate(rows):
            for g in range(stats.mean.shape[1]):
                m,r,m0,r0,mu,ru=(float(v[i,g]) for v in arrays)
                stats_records.append(dict(row,condition=condition,layer=name,group=g,mean=m,rstd=r,full_mean=m0,full_rstd=r0,
                    used_mean=mu,used_rstd=ru,normalized_mean_shift=(m-m0)*r0,
                    log_variance_plus_eps_ratio=2*np.log(r0/r),variance_reconstructed=max(0.,1/r**2-epsilons[name]),
                    self_replay_max_abs=trace.replay_max_abs[name]))
    return records,stats_records


def factorial_table(paired):
    """只对预定义 TRAIN_MEAN 2×2 计算差分，正值表示损失增加。"""
    import pandas as pd
    subset=paired[paired.condition.isin(['TRAIN_MEAN__GN1_FIXED','STAT_ONLY_TRAIN_MEAN','TRAIN_MEAN__NAT'])]
    table=subset.pivot(index=['seed','scope','group','metric'],columns='condition',values='degradation').reset_index()
    table['content']=table['TRAIN_MEAN__GN1_FIXED']
    table['statistics']=table['STAT_ONLY_TRAIN_MEAN']
    table['total']=table['TRAIN_MEAN__NAT']
    table['interaction']=table.total-table.content-table.statistics
    return table.drop(columns=['TRAIN_MEAN__GN1_FIXED','STAT_ONLY_TRAIN_MEAN','TRAIN_MEAN__NAT'])
