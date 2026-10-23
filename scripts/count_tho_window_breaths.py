"""按冻结 THO 波峰语义统计完整180秒窗口，保留全索引与准入分层。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.metrics.task import _detect_peaks
from resp_train.protocols.respiration import canonicalize_numpy


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def count_window(signal):
    # 与训练数据读取器一致，先转换 float32，再由公共规范化函数使用 float64。
    values = np.asarray(signal, dtype=np.float32)
    if values.shape != (18000,) or not np.isfinite(values).all():
        raise ValueError('THO窗口必须为18000点且全部有限')
    band, canonical = canonicalize_numpy(values, fs=100, low_hz=0.05, high_hz=0.70)
    if not np.isfinite(canonical).all():
        raise ValueError('规范化结果包含非有限值')
    peaks = _detect_peaks(canonical, SimpleNamespace(ibi_peak_distance=142))
    energy = float(np.mean((band - band.mean()) ** 2))
    return peaks, energy > 1e-8, energy


def describe(frame):
    values = frame.breath_peak_count.to_numpy()
    return dict(window_n=len(frame), subject_n=int(frame.samp_id.nunique()),
                mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                min=int(values.min()), p05=float(np.quantile(values, 0.05)),
                q25=float(np.quantile(values, 0.25)), median=float(np.median(values)),
                q75=float(np.quantile(values, 0.75)), p95=float(np.quantile(values, 0.95)),
                max=int(values.max()), overlapping_window_sum=int(values.sum()),
                nondynamic_n=int((~frame.target_dynamic).sum()),
                fewer_than_two_peaks_n=int((values < 2).sum()))


def run(config, output, manifest):
    cfg = OmegaConf.load(config)
    if (cfg.window.target_fs != 100 or cfg.window.duration_samples != 18000
            or cfg.loss.band_low_hz != 0.05 or cfg.loss.band_high_hz != 0.70
            or cfg.evaluation.ibi_peak_distance_samples != 142):
        raise ValueError('配置与本次冻结计数口径不一致')
    index = (Path(cfg.data.dataset_root) / cfg.data.index_csv).resolve()
    index_hash = sha_file(index)
    rows = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    if rows.dataset_row_id.duplicated().any() or rows.empty:
        raise ValueError('索引为空或行身份重复')
    if set(rows.split) != {'train', 'val', 'test'}:
        raise ValueError('split身份不符合当前数据集')
    if not rows.target_signal_key.eq('tho_waveform_segment_soft_z').all():
        raise ValueError('THO target key不一致')
    if not np.allclose(rows.window_end_s - rows.window_start_s, 180, rtol=0, atol=1e-8):
        raise ValueError('索引含非180秒窗口')
    OmegaConf.save(cfg, output / 'config.yaml')
    sources, records = [], []
    grouped = rows.groupby(['target_source_npz', 'target_signal_key'], sort=True)
    for number, ((relative, key), group) in enumerate(grouped, 1):
        path = (index.parent / relative).resolve()
        before = path.stat()
        with np.load(path, allow_pickle=False) as archive:
            signal = archive[key]
        if signal.ndim != 1:
            raise ValueError(f'源THO必须为一维：{path}')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'读取期间源文件改变：{path}')
        sources.append(dict(path=str(path), key=key, size_bytes=after.st_size,
                            mtime_ns=after.st_mtime_ns, shape=list(signal.shape), dtype=str(signal.dtype),
                            array_sha256=hashlib.sha256(np.ascontiguousarray(signal).tobytes()).hexdigest()))
        for row in group.itertuples(index=False):
            if row.window_start_sample < 0 or row.window_end_sample > len(signal):
                raise ValueError(f'窗口越界：{row.dataset_row_id}')
            peaks, dynamic, energy = count_window(signal[row.window_start_sample:row.window_end_sample])
            records.append(dict(dataset_row_id=row.dataset_row_id, samp_id=row.samp_id,
                                split=row.split, admitted=bool(row.usable),
                                window_start_s=row.window_start_s, window_end_s=row.window_end_s,
                                breath_peak_count=len(peaks), complete_interpeak_intervals=max(len(peaks)-1, 0),
                                count_rate_bpm=len(peaks)/3, target_dynamic=dynamic, band_energy=energy,
                                peak_indices_json=json.dumps(peaks.tolist()),
                                target_source_npz=relative, target_signal_key=key))
        del signal
        print(f'THO来源 {number}/{len(grouped)}，累计 {len(records)} 窗口', flush=True)
    if sha_file(index) != index_hash:
        raise ValueError('读取期间索引改变')
    frame = pd.DataFrame(records).sort_values('dataset_row_id')
    if len(frame) != len(rows):
        raise ValueError('输出窗口数不一致')
    frame.to_csv(output / 'window_counts.csv', index=False)
    summaries, subjects = [], []
    for scope, subset in [('all_index', frame), ('admitted', frame[frame.admitted]),
                           ('excluded_by_existing_admission', frame[~frame.admitted])]:
        if subset.empty:
            continue
        for split, part in [('all', subset), *subset.groupby('split')]:
            summaries.append(dict(scope=scope, split=split, **describe(part)))
        for (split, subject), part in subset.groupby(['split', 'samp_id']):
            subjects.append(dict(scope=scope, split=split, samp_id=subject, **describe(part)))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(subjects).to_csv(output / 'subject_summary.csv', index=False)
    frame.groupby(['split', 'admitted', 'breath_peak_count']).size().rename('window_n').reset_index().to_csv(
        output / 'count_distribution.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    bins = np.arange(frame.breath_peak_count.min() - 0.5, frame.breath_peak_count.max() + 1.5)
    for ax, (title, subset) in zip(axes, [('All indexed windows', frame), ('Admitted windows', frame[frame.admitted])]):
        for split in ['train', 'val', 'test']:
            ax.hist(subset.loc[subset.split.eq(split), 'breath_peak_count'], bins=bins,
                    histtype='step', label=split, linewidth=1.4)
        ax.set(title=title, xlabel='Detected THO peaks per 180 s', ylabel='Window count')
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'breath_count_distribution.png', dpi=160)
    plt.close(fig)
    write_json(output / 'sources.json', dict(index=dict(path=str(index), sha256=index_hash), arrays=sources))
    manifest.update(status='completed', window_n=len(frame), subject_n=int(frame.samp_id.nunique()),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    outputs={p.name: sha_file(p) for p in sorted(output.iterdir()) if p.name != 'manifest.json'})
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/tho_research_v2.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dependencies = [Path(__file__), ROOT / 'resp_train/metrics/task.py',
                    ROOT / 'resp_train/protocols/respiration.py', ROOT / 'resp_train/data/research_v2.py',
                    ROOT / 'docs/experiments/tho_breath_count_v1_20261019.md']
    manifest = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                    implementation={str(p.relative_to(ROOT)): sha_file(p) for p in dependencies},
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
                    interpretation='逐180秒窗口的THO正峰数；窗口间重叠，计数总和未去重；算法估计而非人工真值')
    write_json(args.output / 'manifest.json', manifest)
    try:
        run(args.config, args.output, manifest)
    except BaseException:
        manifest.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
