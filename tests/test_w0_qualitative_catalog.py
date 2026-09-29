"""按需分析入口的合成数据测试。"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resp_train.paper_evidence.w0_qualitative_catalog import Bundle, create_index, select_frame, select_cases, selection_rows, table_page
from resp_train.paper_evidence.w0_test_qualitative import PRIMARY, SEED
from resp_train.paper_evidence.w0_test_qualitative_runtime import finish


def candidates():
    return pd.DataFrame({"dataset_row_id": [1,2,3,4,5,6], "samp_id": [10,10,10,20,20,20],
        "envelope_target_stratum": ['low']*6, "target_envelope_modulation": [.1,.2,.3,.1,.2,.3],
        "transient_motion_ratio": [0.]*6, "posture_transition_ratio": [0.]*6,
        "W0_lag_aware_signed_pcc": [.99,.2,.8,.7,.4,.9]})


def test_selection_subject_balance_and_no_prediction_ranking():
    frame = candidates()
    _, selected = select_frame(frame, quality='zero-markers', strata=['low'], subjects=None, per_subject=1)
    assert selected.dataset_row_id.tolist() == [2,5]
    frame.W0_lag_aware_signed_pcc = frame.W0_lag_aware_signed_pcc[::-1].to_numpy()
    _, second = select_frame(frame, quality='zero-markers', strata=['low'], subjects=None, per_subject=1)
    assert second.dataset_row_id.tolist() == [2,5]
    frame.loc[1, 'transient_motion_ratio'] = .1
    _, selected = select_frame(frame, quality='zero-markers', strata=['low'], subjects=[10], per_subject=1)
    assert selected.dataset_row_id.tolist() == [1]
    with pytest.raises(ValueError, match='为空'):
        select_frame(frame, quality='zero-markers', strata=['high'], subjects=None, per_subject=1)


def test_index_selection_and_source_binding(tmp_path):
    export = tmp_path/'export'; export.mkdir()
    base = candidates()
    rows = base[['dataset_row_id','samp_id','transient_motion_ratio','posture_transition_ratio']].copy()
    rows['window_start_s'] = np.arange(6)*30
    rows['window_end_s'] = rows.window_start_s+180
    rows.to_csv(export/'test_rows.csv',index=False)
    index = rows.drop(columns=['transient_motion_ratio','posture_transition_ratio']).copy()
    index['file'] = index.dataset_row_id.map(lambda i:f'windows/row_{i}.npz')
    index.to_csv(export/'window_index.csv',index=False)
    m = base[['dataset_row_id','target_envelope_modulation','envelope_target_stratum']].copy()
    m['method'] = 'W0'
    m.to_csv(export/'metrics.csv',index=False)
    finish(export, {'phase':'export','seed':SEED})
    idx=create_index(export,tmp_path/'index',figures=None,e4_root=None,command='synthetic')
    assert len(Bundle(idx,'analysis_index').csv('windows.csv')) == 6
    cases=select_cases(idx,tmp_path/'cases',quality='zero-markers',strata=['low'],subjects=None,per_subject=1,command='synthetic')
    assert selection_rows(cases,export) == [2,5]
    assert not json.loads((cases/'selection.json').read_text())['prediction_metrics_used_for_selection']
    with pytest.raises(FileExistsError):
        create_index(export,idx,figures=None,e4_root=None,command='synthetic')
    (export/'artifact_manifest.json').write_text((export/'artifact_manifest.json').read_text()+' ')
    with pytest.raises(ValueError, match='不匹配'):
        selection_rows(cases,export)


def test_html_escapes_data_and_has_filters():
    frame=candidates(); frame['label']='<script>unsafe</script>'
    page=table_page(frame,'索引')
    assert '&lt;script&gt;unsafe&lt;/script&gt;' in page
    assert 'data-col=' in page and 'sortRows' in page


def test_intervention_cli_requires_confirmation(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/export_w0_test_qualitative.py'
    result=subprocess.run([sys.executable,str(script),'intervene-r3','--source','missing',
                           '--cases','missing','--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode == 2
    assert '--confirm-research-test-intervention' in result.stderr
    assert not (tmp_path/'out').exists()


def test_four_conditions_capture_and_gn_controls():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from resp_train.crd.tf_v1_model import CwtBranch
    from resp_train.paper_evidence.w0_qualitative_intervention import CONDITIONS, condition_outputs
    from resp_train.paper_evidence.e4_r3_norm import GN_NAMES, make_shifts
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.tf_variant='crd_tf102_w'
            self.branches=nn.ModuleDict({'w':CwtBranch()})
            self.base=nn.Module()
            self.base.local_blocks=nn.ModuleList([nn.Identity() for _ in range(6)])
            self.base.refinement=nn.Identity()
            with torch.no_grad():
                self.branches['w'].final_projection.weight.fill_(.015)
                for block in self.branches['w'].temporal.blocks:
                    block.project.weight.fill_(.01)
        def forward(self,x,*,tf):
            z=F.interpolate(x,size=1800).expand(-1,96,-1).transpose(1,2)
            for block in self.base.local_blocks: z=block(z)
            gamma,beta=self.branches['w'](tf)
            zp=self.base.refinement(z.transpose(1,2)*(1+.5*torch.tanh(gamma))+.5*torch.tanh(beta))
            return {'waveform':F.interpolate(zp.mean(1,keepdim=True),size=18000)}
    try:
        torch.manual_seed(71)
        model=Model().eval()
        x=torch.randn(1,1,18000); w=torch.rand(1,97,360); original=w.clone()
        results=list(condition_outputs(model,x,w,torch.from_numpy(make_shifts(1))))
        assert tuple(c for c,_,_ in results)==CONDITIONS
        np.testing.assert_allclose(results[0][1]['prediction'],results[2][1]['prediction'],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(w,original,rtol=0,atol=0)
        for condition,arrays,stats in results:
            np.testing.assert_array_equal(arrays['cwt_w'][:,:73],w.numpy()[:,:73])
            assert arrays['z_prime'].shape==(1,96,1800)
            if condition.endswith('ALL_W_GN_FIXED'):
                for layer in GN_NAMES:
                    np.testing.assert_array_equal(stats[layer]['used_mean'],stats[layer]['full_mean'])
                    np.testing.assert_array_equal(stats[layer]['used_rstd'],stats[layer]['full_rstd'])
        assert not np.array_equal(results[0][1]['cwt_w'],results[1][1]['cwt_w'])
        assert not np.array_equal(results[0][1]['z_prime'],results[1][1]['z_prime'])
    finally:
        torch.set_num_threads(previous)


def test_intervention_offline_plot(tmp_path):
    from resp_train.paper_evidence.w0_qualitative_intervention import CONDITIONS, render_intervention
    from resp_train.paper_evidence.w0_test_qualitative import save_arrays
    source=tmp_path/'intervention'
    (source/'windows/row_7').mkdir(parents=True)
    t=np.arange(18000)/100
    target=np.sin(2*np.pi*.25*t)
    records=[]
    for condition in CONDITIONS:
        save_arrays(source/f'windows/row_7/{condition}.npz',{
            'dataset_row_id':np.asarray(7), 'prediction':target[None], 'target':target,
            'cwt_w':np.ones((97,360)), 'log_rms_envelope':np.zeros(35),
            'target_log_rms_envelope':np.zeros(35), 'relative_feature_change':np.zeros(1800),
            'prediction_minus_full':np.zeros((1,18000)),
        })
        records.append({'dataset_row_id':7,'condition':condition,**{key:.1 for key in PRIMARY}})
    pd.DataFrame(records).to_csv(source/'metrics.csv',index=False)
    finish(source,{'phase':'r3_intervention','rows':[7]})
    output=render_intervention(source,tmp_path/'plots',row_ids=None,zoom=(30,60),command='synthetic')
    assert (output/'figures/row_7.png').is_file()
    assert (output/'index.html').is_file()
    with pytest.raises(FileExistsError):
        render_intervention(source,output,row_ids=None,zoom=None,command='synthetic')


def test_intervention_metric_metadata_accepts_pandas_strings():
    from resp_train.paper_evidence.w0_qualitative_intervention import metric_metadata
    from resp_train.metrics.task import _metadata_record

    for dtype in (object, "string"):
        frame = pd.DataFrame({"dataset_row_id": [7, 9], "samp_id": [10, 20],
                              "split": pd.Series(["test", "test"], dtype=dtype),
                              "coupling_state_id": [1, 2]}).set_index("dataset_row_id", drop=False)
        metadata = metric_metadata(frame, [9, 7])
        record = _metadata_record(metadata, index=0, method="FULL__NAT")
        assert record == {"method": "FULL__NAT", "dataset_row_id": 9,
                          "samp_id": 20, "split": "test", "coupling_state_id": 2}
