"""固定 W0 seed 的详细测试结果导出和离线绘图入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    export_parser = sub.add_parser("export", help="真实测试集单次 forward + F0/IEWT 导出")
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--device", default="cuda:0")
    export_parser.add_argument("--confirm-research-test-export", action="store_true")
    finalize_parser = sub.add_parser("finalize", help="检查已有完整导出并生成离线索引与完成清单")
    finalize_parser.add_argument("--source", type=Path, required=True)
    render_parser = sub.add_parser("render", help="仅从导出文件重绘")
    render_parser.add_argument("--source", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--rows", nargs="+", type=int, help="省略时绘制全部窗口")
    render_parser.add_argument("--cases", type=Path, help="已完成的案例选择目录")
    render_parser.add_argument("--channels", nargs="+", type=int, help="conditioning 热图的 latent 通道编号，0–95")
    render_parser.add_argument("--views", nargs="+", choices=("waveforms", "conditioning", "trajectories"),
                               default=["waveforms", "conditioning", "trajectories"])
    render_parser.add_argument("--zoom", nargs=2, type=float, metavar=("START_S", "END_S"))
    index_parser = sub.add_parser("index", help="合并窗口质量/指标与 E4 证据，生成筛选页面")
    index_parser.add_argument("--source", type=Path, required=True)
    index_parser.add_argument("--output", type=Path, required=True)
    index_parser.add_argument("--figures", type=Path)
    index_parser.add_argument("--e4-root", type=Path, default=ROOT / "runs/e4_r3_temporal_normalization_v1")
    index_parser.add_argument("--skip-e4", action="store_true")
    select_parser = sub.add_parser("select-cases", help="按主体与参考包络分层保存案例清单")
    select_parser.add_argument("--index", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--quality", choices=("zero-markers", "all"), default="zero-markers")
    select_parser.add_argument("--strata", nargs="+", choices=("low", "medium", "high"), default=["low", "medium"])
    select_parser.add_argument("--subjects", nargs="+", type=int)
    select_parser.add_argument("--per-subject", type=int, default=1, help="每主体×分层的案例数")
    intervention_parser = sub.add_parser("intervene-r3", help="固定四条件的所选测试窗口 GPU 干预")
    intervention_parser.add_argument("--source", type=Path, required=True)
    intervention_parser.add_argument("--cases", type=Path, required=True)
    intervention_parser.add_argument("--output", type=Path, required=True)
    intervention_parser.add_argument("--device", default="cuda:0")
    intervention_parser.add_argument("--confirm-research-test-intervention", action="store_true")
    intervention_render_parser = sub.add_parser("render-intervention", help="从保存干预结果生成机制图")
    intervention_render_parser.add_argument("--source", type=Path, required=True)
    intervention_render_parser.add_argument("--output", type=Path, required=True)
    intervention_render_parser.add_argument("--rows", nargs="+", type=int)
    intervention_render_parser.add_argument("--zoom", nargs=2, type=float, metavar=("START_S", "END_S"))
    args = parser.parse_args()
    if args.phase == "export" and not args.confirm_research_test_export:
        parser.error("真实 test 导出必须提供 --confirm-research-test-export")
    if args.phase == "intervene-r3" and not args.confirm_research_test_intervention:
        parser.error("真实 test 干预必须提供 --confirm-research-test-intervention")
    from resp_train.paper_evidence.w0_test_qualitative_runtime import export, finalize, render
    command = shlex.join([sys.executable, *sys.argv])
    if args.phase == "export":
        result = export(args.output, device=args.device, command=command)
    elif args.phase == "finalize":
        result = finalize(args.source, command=command)
    elif args.phase == "render":
        result = render(args.source, args.output, row_ids=args.rows,
                        zoom=tuple(args.zoom) if args.zoom else None, command=command,
                        views=tuple(args.views), cases=args.cases,
                        channels=tuple(args.channels) if args.channels else None)
    elif args.phase == "index":
        from resp_train.paper_evidence.w0_qualitative_catalog import create_index
        result = create_index(args.source, args.output, figures=args.figures,
                              e4_root=None if args.skip_e4 else args.e4_root, command=command)
    elif args.phase == "select-cases":
        from resp_train.paper_evidence.w0_qualitative_catalog import select_cases
        result = select_cases(args.index, args.output, quality=args.quality, strata=args.strata,
                              subjects=args.subjects, per_subject=args.per_subject, command=command)
    elif args.phase == "intervene-r3":
        from resp_train.paper_evidence.w0_qualitative_intervention import intervene
        result = intervene(args.source, args.cases, args.output, device=args.device, command=command)
    elif args.phase == "render-intervention":
        from resp_train.paper_evidence.w0_qualitative_intervention import render_intervention
        result = render_intervention(args.source, args.output, row_ids=args.rows,
                                     zoom=tuple(args.zoom) if args.zoom else None, command=command)
    print(result)


if __name__ == "__main__":
    main()
