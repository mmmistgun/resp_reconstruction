"""启动 BCG-CWT train/val 训练，或仅解析配置。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from resp_train.bcg_cwt_training import load_training_config, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/bcg_cwt_patch_mamba_v1/train.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--print-config", action="store_true", help="仅解析配置，不读取数据或创建运行目录")
    parser.add_argument("--resume", action="store_true", help="从同一运行目录最后一个完整 epoch 恢复")
    args = parser.parse_args()
    cfg = load_training_config(args.config, args.overrides)
    if args.resume:
        cfg.training.resume = True
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    else:
        print(train(cfg))


if __name__ == "__main__":
    main()
