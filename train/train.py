"""TRN-04 전이 학습 / TRN-05 모델 크기별 비교용 학습 스크립트.

configs/default.yaml 의 train 설정으로 사전학습 모델들을 같은 조건에서 학습합니다.
결과는 runs/<모델 이름>/ 에 저장되고, 재현용 기록(config.yaml)을 함께 남깁니다(TRN-07).

사용 예 (저장소 루트에서 실행):
    python train/train.py                       # 설정 파일의 모델 4개 모두 학습
    python train/train.py --models yolo11s      # 한 모델만 학습
    python train/train.py --models yolov8s --epochs 1 --device cpu --name-suffix _smoke   # 동작 확인용
"""

import argparse
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]          # 저장소 루트
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="PPE 탐지 모델 전이 학습 (TRN-04, TRN-05)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="설정 파일 경로")
    p.add_argument("--models", nargs="+", help="학습할 모델 (기본: 설정 파일의 train.models)")
    p.add_argument("--epochs", type=int, help="설정값 대신 쓸 epoch 수 (동작 확인용)")
    p.add_argument("--batch", type=int, help="설정값 대신 쓸 batch 크기")
    p.add_argument("--device", help="설정값 대신 쓸 장치 (예: 0, cpu)")
    p.add_argument("--name-suffix", default="", help="실험 이름 뒤에 붙일 문자열 (예: _smoke)")
    return p.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg or "train" not in cfg:
        sys.exit(f"[오류] {path} 에 train 설정이 없습니다.")
    return cfg["train"]


def check_dataset(data_yaml, expected_names):
    """data.yaml이 있고 클래스 순서가 고정 순서와 같은지 학습 전에 확인합니다."""
    if not data_yaml.is_file():
        sys.exit(f"[오류] data.yaml을 찾을 수 없습니다: {data_yaml}")
    with open(data_yaml, encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if names != expected_names:
        sys.exit(f"[오류] 클래스 순서가 다릅니다.\n  data.yaml: {names}\n  설정 파일: {expected_names}")


def is_oom(err):
    return "out of memory" in str(err).lower()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def environment_info():
    import torch
    import ultralytics

    # torch.__version__ 는 str을 상속한 TorchVersion 객체라 yaml.safe_dump가 저장하지 못하므로 str()로 바꿉니다.
    info = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": str(torch.__version__),
        "ultralytics": str(ultralytics.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda"] = str(torch.version.cuda)
        info["gpu"] = str(torch.cuda.get_device_name(0))
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return info


def train_one(model_name, cfg, data_yaml, name):
    """한 모델을 학습합니다. GPU 메모리가 부족하면 batch를 절반으로 줄여 다시 시도합니다(TRN-04 예외 처리)."""
    from ultralytics import YOLO

    batch = cfg["batch"]
    while True:
        print(f"\n===== {model_name} 학습 시작 (batch={batch}) =====")
        model = YOLO(f"{model_name}.pt")                # 사전학습(COCO) 가중치에서 시작
        try:
            model.train(
                data=str(data_yaml),
                imgsz=cfg["imgsz"],
                epochs=cfg["epochs"],
                batch=batch,
                seed=cfg["seed"],
                deterministic=cfg["deterministic"],
                device=cfg["device"],
                project=str(ROOT / cfg["project"]),
                name=name,
                exist_ok=False,
            )
            return model.trainer.save_dir, batch
        except Exception as e:
            if is_oom(e) and batch > 1:
                print(f"[경고] GPU 메모리 부족: batch {batch} -> {batch // 2} 로 다시 시도합니다.")
                batch //= 2
                import torch
                torch.cuda.empty_cache()
                continue
            raise


def save_record(save_dir, model_name, cfg, batch_used, env):
    """재현용 기록을 runs/<실험이름>/config.yaml 에 남깁니다(TRN-07)."""
    record = {
        "model": model_name,
        "dataset": cfg["dataset_version"],
        "data_yaml": cfg["data"],
        "names": cfg["names"],
        "imgsz": cfg["imgsz"],
        "epochs": cfg["epochs"],
        "batch_config": cfg["batch"],
        "batch_used": batch_used,
        "seed": cfg["seed"],
        "deterministic": cfg["deterministic"],
        "augmentation": "Ultralytics 기본값 (전체 인자는 같은 폴더의 args.yaml)",
        "git_commit": git_commit(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "environment": env,
    }
    with open(Path(save_dir) / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, allow_unicode=True, sort_keys=False)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    for key in ("epochs", "batch", "device"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    models = args.models or cfg["models"]

    data_yaml = (ROOT / cfg["data"]).resolve()
    check_dataset(data_yaml, cfg["names"])

    env = environment_info()
    print("실행 환경:", env)
    if str(cfg["device"]) != "cpu" and not env["cuda_available"]:
        sys.exit("[오류] CUDA를 쓸 수 없습니다. GPU PC에서 실행하거나, 동작 확인만 할 때는 --device cpu 를 붙이세요.")

    results = []
    for m in models:
        save_dir, batch_used = train_one(m, cfg, data_yaml, name=m + args.name_suffix)
        save_record(save_dir, m, cfg, batch_used, env)
        results.append((m, save_dir, batch_used))

    print("\n===== 학습 완료 =====")
    for m, save_dir, batch_used in results:
        print(f"{m:10s} batch={batch_used:<3d} -> {Path(save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":          # Windows에서 데이터 로더 워커가 이 파일을 다시 실행하므로 필요
    main()
