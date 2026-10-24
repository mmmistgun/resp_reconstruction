"""核验M4-S1总loss选点及精度修订；所有派生产物写入新的输出目录。"""


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confirm-research-test', action='store_true')
    args = parser.parse_args()
    if not args.confirm_research_test:
        parser.error('读取research-test产物需要 --confirm-research-test 及当次授权')
    if not __debug__:
        parser.error('核验依赖assert，禁止使用Python -O或PYTHONOPTIMIZE')
    T = args.root.resolve()
    C = args.output.resolve()
    # 输出必须独立；任何已有目录（包括失败产物）均不能覆盖。
    C.mkdir(parents=True, exist_ok=False)
    from pathlib import Path
    import sys
    from datetime import datetime, timezone
    import numpy as np
    import pandas as pd

    R=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(R))
    from scripts import eval_m4_s1_loss_selected_test as task

    with task.configured(T) as (rt,lock):
        previous=Path(lock['previous_root'])
        training=Path(lock['training_session']['path']).parent
        pipeline=task.read(T/'pipeline_completed.json')
        assert pipeline['status']=='completed' and pipeline['planned']==84
        assert task.read(T/'workers_exit.json')['returncodes']==[0,0]
        assert not list((T/'evaluation').glob('*/attempt_*/failed.json'))
        assert len(list((T/'evaluation').glob('*/attempt_*/receipt.json')))==84
        data,metadata=rt.get_data(T)
        olddata=task.previous_stage(rt,previous,'data')
        rt.data_ops.verify_sources(metadata)
        assert rt.io.identity(data/'test_reference.npy')==rt.io.identity(olddata/'test_reference.npy')
        rows=pd.read_csv(data/'test_rows.csv')
        pd.testing.assert_frame_equal(rows,pd.read_csv(olddata/'test_rows.csv'))
        dev=set()
        for split in ('train','val'):
            dev.update(pd.read_csv(T/f'{split}_rows.csv').samp_id.astype(int))
        rt.check_rows(rows,dev)
        caches={}
        for name,rep in lock['representations'].items():
            path=rt.completed(T,'cache',representation=name)
            old=task.previous_stage(rt,previous,'cache',representation=name)
            assert rt.io.identity(path/'test_w.npy')==rt.io.identity(old/'test_w.npy')
            reader=rt.TestReader(path,rep,rows.dataset_row_id.to_numpy(np.int64))
            for i in range(0,len(rows),32):assert np.isfinite(reader.values[i:i+32]).all()
            caches[name]=rt.source(path/'manifest.json')
        frames,oldframes,paths,cell_rows=[],[],[],[]
        for entry in lock['entries']:
            cell=training/'cells'/entry['cell']
            rt.verify_source(entry['completion']);receipt=task.read(cell/'completed.json')
            history=[]
            for path in sorted((cell/'epochs').glob('*/record.json')):
                rt.io.verify(path,receipt['artifacts'][str(path.relative_to(cell))])
                history.append(task.read(path))
            assert len(history)==receipt['completed_epochs']
            picked=task.select_loss(history)
            assert picked['epoch']==entry['epoch'] and picked['validation']['loss']==entry['selected_validation_loss']
            old_record=history[entry['previous_epoch']-1]
            assert old_record['validation']['loss']==entry['previous_validation_loss']
            assert old_record['val_local_rr_mae']==entry['previous_validation_local_rr']
            assert picked['val_local_rr_mae']==entry['selected_validation_local_rr']
            for field in ('config','checkpoint','previous_checkpoint'):rt.verify_source(entry[field])
            path=task.result_path(rt,T,entry)
            old=task.previous_stage(rt,previous,'evaluation',cell=entry['cell'])
            result=task.read(path/'result.json')
            assert (result['cell'],result['epoch'],result['parameters'],result['count'],result['subjects'])==(entry['cell'],entry['epoch'],entry['parameters'],2310,8)
            assert entry['reuse_previous']==(entry['checkpoint']==entry['previous_checkpoint'])
            assert (path==old)==entry['reuse_previous']
            reference=task.read(path/'reference_source.json')
            assert {k:reference[k] for k in ('sha256','size_bytes')}==rt.io.identity(data/'test_reference.npy')
            env=task.read(path/'environment.json')
            assert env['dependencies']==lock['dependencies'] and env['gpu']=='NVIDIA GeForce RTX 4070 Ti SUPER'
            cfg=rt.OmegaConf.load(entry['config']['path'])
            cfg.training.device=env['device'];cfg.outputs.run_root=str(path)
            assert rt.OmegaConf.to_container(cfg,resolve=True)==rt.OmegaConf.to_container(rt.OmegaConf.load(path/'evaluation_config.yaml'),resolve=True)
            pd.testing.assert_frame_equal(pd.read_csv(path/'test_rows.csv'),rows)
            values=np.load(path/'prediction.npy',mmap_mode='r',allow_pickle=False)
            assert values.shape==(2310,1,18000) and values.dtype==np.float32
            for i in range(0,len(values),32):assert np.isfinite(values[i:i+32]).all()
            newframe,oldframe=pd.read_csv(path/'metrics.csv'),pd.read_csv(old/'metrics.csv')
            for frame in (newframe,oldframe):
                assert frame.arm.eq(entry['arm']).all() and frame.seed.eq(entry['seed']).all()
                rt.sf.validate_metrics(frame,rows)
            eligible=['dataset_row_id','samp_id','split']+sorted(c for c in newframe if c.endswith('_target_eligible') or c=='local_rr_target_eligible_windows')
            pd.testing.assert_frame_equal(newframe[eligible],oldframe[eligible])
            if entry['reuse_previous']:pd.testing.assert_frame_equal(newframe,oldframe)
            frames.append(newframe);oldframes.append(oldframe);paths.append(path)
            record={k:entry[k] for k in ('cell','arm','seed','epoch','previous_epoch','selected_validation_loss','previous_validation_loss','selected_validation_local_rr','previous_validation_local_rr','reuse_previous')}
            for metric in rt.sf.PRIMARY:
                newmean=float(newframe.loc[rt.sf._metric_mask(newframe,metric),metric].mean())
                oldmean=float(oldframe.loc[rt.sf._metric_mask(oldframe,metric),metric].mean())
                record.update({metric+'_loss':newmean,metric+'_local_rr':oldmean,metric+'_delta':newmean-oldmean})
            cell_rows.append(record)
            print('verified',entry['cell'],flush=True)
        summary=rt.completed(T,'summary');oldsummary=task.previous_stage(rt,previous,'summary')
        sources=task.read(summary/'sources.json')
        assert sources['evaluations']==[rt.source(p/'manifest.json') for p in paths]
        assert sources['allowlist']==rt.source(T/'allowlist.json')
        assert sources['previous_summary']==rt.source(oldsummary/'manifest.json')
        assert (sources['new_evaluations'],sources['reused_evaluations'])==(82,2)
        tables=rt.summary_tables(pd.concat(frames,ignore_index=True))
        oldtables=rt.summary_tables(pd.concat(oldframes,ignore_index=True))
        for name,table in tables.items():
            pd.testing.assert_frame_equal(pd.read_csv(summary/f'{name}.csv'),table,check_dtype=False,rtol=1e-10,atol=1e-12)
            pd.testing.assert_frame_equal(pd.read_csv(oldsummary/f'{name}.csv'),oldtables[name],check_dtype=False,rtol=1e-10,atol=1e-12)
        corrections=[]
        reused={(e['arm'],e['seed']) for e in lock['entries'] if e['reuse_previous']}
        for prefix in ('','subject_macro_'):
            means='per_seed' if not prefix else 'subject_macro'
            new=tables[means];old=oldtables[means]
            paired=new[['arm','seed','metric','mean']].merge(old[['arm','seed','metric','mean']],on=['arm','seed','metric'],suffixes=('_loss','_local_rr'),validate='one_to_one')
            paired['delta']=paired.mean_loss-paired.mean_local_rr
            paired['improvement']=np.where(paired.metric==rt.sf.PCC,paired.delta,-paired.delta)
            paired['group']='checkpoint_selector';paired['comparison']=paired.arm+'_loss_vs_local_rr';paired['reference']=paired.arm
            pd.testing.assert_frame_equal(pd.read_csv(summary/f'{prefix}selector_paired.csv'),paired,check_dtype=False,rtol=1e-10,atol=1e-12)
            original= pd.read_csv(summary/f'{prefix}selector_summary.csv')
            verified=rt.stats.pair_summary(paired)
            directions=['improved_seeds','worsened_seeds','tied_seeds']
            pd.testing.assert_frame_equal(original.drop(columns=directions),verified.drop(columns=directions),check_dtype=False,rtol=1e-10,atol=1e-12)
            for row in paired.itertuples():
                if (row.arm,row.seed) in reused:assert row.delta==0 and row.improvement==0
            for index in original.index:
                if not original.loc[index,directions].equals(verified.loc[index,directions]):
                    assert original.loc[index,'arm'] in {arm for arm,_ in reused}
                    assert verified.loc[index,'tied_seeds']==original.loc[index,'tied_seeds']+1
                    assert abs(original.loc[index,'delta_mean']-verified.loc[index,'delta_mean'])<1e-14
                    corrections.append({'aggregation':prefix or 'window_weighted','arm':original.loc[index,'arm'],
                        'metric':original.loc[index,'metric'],**{f'{col}_original':int(original.loc[index,col]) for col in directions},
                        **{f'{col}_verified':int(verified.loc[index,col]) for col in directions}})
            paired.to_csv(C/f'{prefix}selector_paired_verified.csv',index=False)
            verified.to_csv(C/f'{prefix}selector_summary_verified.csv',index=False)
        pd.DataFrame(corrections).to_csv(C/'precision_direction_corrections.csv',index=False)
        pd.DataFrame(cell_rows).to_csv(C/'cell_comparison_84.csv',index=False)
        subject=tables['per_subject'].merge(oldtables['per_subject'],on=['arm','seed','samp_id','metric'],suffixes=('_loss','_local_rr'),validate='one_to_one')
        assert subject.eligible_n_loss.equals(subject.eligible_n_local_rr)
        subject['delta']=subject.mean_loss-subject.mean_local_rr
        subject['improvement']=np.where(subject.metric==rt.sf.PCC,subject.delta,-subject.delta)
        subject['weight']=subject.eligible_n_loss/2310
        subject['weighted_contribution']=subject.delta*subject.weight
        subject.to_csv(C/'subject_selector_paired.csv',index=False)
        subject.groupby(['arm','samp_id','metric']).agg(delta_mean=('delta','mean'),delta_sd=('delta','std'),
            weighted_contribution=('weighted_contribution','mean'),improved_seeds=('improvement',lambda x:int((x>0).sum())),
            worsened_seeds=('improvement',lambda x:int((x<0).sum()))).reset_index().to_csv(C/'subject_selector_summary.csv',index=False)
        rt.io.write_json(C/'audit.json',{'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'protocol':task.PROTOCOL,
            'cells':84,'new_evaluations':82,'reused_cells':[e['cell'] for e in lock['entries'] if e['reuse_previous']],
            'windows_per_cell':2310,'subjects':rows.groupby('samp_id').size().to_dict(),
            'metric_n_ranges':tables['per_seed'].groupby('metric')['n'].agg(['min','max']).to_dict('index'),
            'allowlist':rt.source(T/'allowlist.json'),'summary':rt.source(summary/'manifest.json'),
            'previous_summary':rt.source(oldsummary/'manifest.json'),'verified_native_tables':17,'verified_selector_tables':4,
            'precision_corrections':corrections,'precision_note':rt.source(R/'docs/experiments/m4_s1_loss_selected_test_precision_note_20261024.md'),
            'audit_script':rt.source(__file__),'cell_comparison':rt.source(C/'cell_comparison_84.csv'),
            'subject_comparison':rt.source(C/'subject_selector_summary.csv'),'default_model_changed':False})
        rt.io.write_json(C/'audit_receipt.json',rt.io.identity(C/'audit.json'))
        print('AUDIT PASSED: 84 cells, 21 tables and previous comparison verified',flush=True)


if __name__ == '__main__':
    main()
