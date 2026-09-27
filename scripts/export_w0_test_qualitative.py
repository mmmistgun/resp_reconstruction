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
    render_parser = sub.add_parser("render", help="仅从导出文件重绘")
    render_parser.add_argument("--source", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--rows", nargs="+", type=int, help="省略时绘制全部窗口")
    render_parser.add_argument("--zoom", nargs=2, type=float, metavar=("START_S", "END_S"))
    args = parser.parse_args()
    if args.phase == "export" and not args.confirm_research_test_export:
        parser.error("真实 test 导出必须提供 --confirm-research-test-export")
    from resp_train.paper_evidence.w0_test_qualitative_runtime import export, render
    command = shlex.join([sys.executable, *sys.argv])
    if args.phase == "export":
        result = export(args.output, device=args.device, command=command)
    else:
        result = render(args.source, args.output, row_ids=args.rows,
                        zoom=tuple(args.zoom) if args.zoom else None, command=command)
    print(result)


if __name__ == "__main__":
    main()
