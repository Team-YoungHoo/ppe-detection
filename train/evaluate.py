"""EVL-01 탐지 성능 평가 스크립트.

train/train.py 로 학습한 모델(runs/<모델 이름>/weights/best.pt)을 valid·test에서 평가해
전체 mAP와 클래스별 AP·Precision·Recall, 클래스별 정답 박스 수를 표로 남깁니다.
결과: runs/eval/results.csv, runs/eval/results.md (노션에 붙여넣기용), runs/eval/<모델>_<분할>/ (그래프)

사용 예 (저장소 루트에서 실행):
    python train/evaluate.py                          # 설정 파일의 모델 4개 모두 평가
    python train/evaluate.py --models yolo11s         # 한 모델만 평가
    python train/evaluate.py --models yolov8s --name-suffix _smoke --device cpu   # 동작 확인용 학습 결과 평가
    python train/evaluate.py --weights path/to/best.pt                            # 가중치 파일 직접 지정

참고: Precision·Recall은 Ultralytics가 F1이 가장 높은 신뢰도에서 계산한 값입니다.
      실제 C++ 실행 시의 신뢰도 임계값(DET-03)에서의 값과는 다를 수 있습니다.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

from train import DEFAULT_CONFIG, ROOT, check_dataset, environment_info  # 같은 폴더의 train.py

SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}   # data.yaml 키 -> 표에 쓸 이름


def parse_args():
    p = argparse.ArgumentParser(description="PPE 탐지 모델 성능 평가 (EVL-01)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="설정 파일 경로")
    p.add_argument("--models", nargs="+", help="평가할 모델 (기본: 설정 파일의 train.models)")
    p.add_argument("--name-suffix", default="", help="학습 때 쓴 실험 이름 접미사 (예: _smoke)")
    p.add_argument("--weights", nargs="+", help="모델 이름 대신 가중치 파일 경로를 직접 지정")
    p.add_argument("--splits", nargs="+", help="평가할 분할 (기본: 설정 파일의 eval.splits)")
    p.add_argument("--device", help="설정값 대신 쓸 장치 (예: 0, cpu)")
    return p.parse_args()


def load_configs(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg or "train" not in cfg or "eval" not in cfg:
        sys.exit(f"[오류] {path} 에 train 또는 eval 설정이 없습니다.")
    return cfg["train"], cfg["eval"]


def iou_index(iou_thr):
    """Ultralytics는 IoU 0.50~0.95(0.05 단위) 10개 기준의 AP를 계산합니다. 그중 몇 번째인지 구합니다."""
    idx = round((iou_thr - 0.5) / 0.05)
    if not (0 <= idx <= 9 and abs(0.5 + idx * 0.05 - iou_thr) < 1e-6):
        sys.exit(f"[오류] eval.iou_thr={iou_thr} 는 지원하지 않습니다. 0.50~0.95 사이 0.05 단위로 적으세요.")
    return idx


def count_boxes(data_yaml, split, nc):
    """분할의 클래스별 정답 박스 수. 샘플이 적은 클래스의 AP를 해석할 때 함께 봅니다."""
    with open(data_yaml, encoding="utf-8") as f:
        rel = yaml.safe_load(f)[split]
    lbl_dir = data_yaml.parent / Path(rel).parent / "labels"
    cnt = Counter()
    for lb in lbl_dir.glob("*.txt"):
        for line in lb.read_text().splitlines():
            if line.strip():
                cnt[int(line.split()[0])] += 1
    return [cnt[i] for i in range(nc)]


def weight_list(args, tcfg):
    """(표에 쓸 이름, best.pt 경로) 목록. 없는 파일은 경고하고 건너뜁니다."""
    if args.weights:
        pairs = [(Path(w).resolve().parents[1].name, Path(w)) for w in args.weights]
    else:
        names = [m + args.name_suffix for m in (args.models or tcfg["models"])]
        pairs = [(n, ROOT / tcfg["project"] / n / "weights" / "best.pt") for n in names]
    found = []
    for name, w in pairs:
        if w.is_file():
            found.append((name, w))
        else:
            print(f"[경고] 가중치가 없어 건너뜁니다: {w}")
    if not found:
        sys.exit("[오류] 평가할 가중치가 없습니다. train/train.py 로 먼저 학습하세요.")
    return found


def evaluate(name, weights, split, tcfg, ecfg, data_yaml, idx):
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(
        data=str(data_yaml),
        split=split,
        imgsz=tcfg["imgsz"],
        batch=tcfg["batch"],
        device=tcfg["device"],
        project=str(ROOT / ecfg["output"]),
        name=f"{name}_{SPLIT_DIRS[split]}",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    box = m.box
    nc = len(tcfg["names"])
    ap = [None] * nc
    p = [None] * nc
    r = [None] * nc
    # ap_class_index: 이 분할에 정답 박스가 있는 클래스만 들어 있습니다. 없는 클래스는 N/A(EVL-01 예외 처리)
    for row, c in enumerate(box.ap_class_index):
        ap[int(c)] = float(box.all_ap[row, idx])
        p[int(c)] = float(box.p[row])
        r[int(c)] = float(box.r[row])
    present = [a for a in ap if a is not None]
    map_thr = sum(present) / len(present) if present else None
    return {"map": map_thr, "map50_95": float(box.map), "ap": ap, "p": p, "r": r}


def fmt(v):
    return "N/A" if v is None else f"{v:.3f}"


def main():
    args = parse_args()
    tcfg, ecfg = load_configs(args.config)
    if args.device is not None:
        tcfg["device"] = args.device
    splits = args.splits or ecfg["splits"]
    for s in splits:
        if s not in SPLIT_DIRS:
            sys.exit(f"[오류] 알 수 없는 분할: {s} (가능: {list(SPLIT_DIRS)})")

    names = tcfg["names"]
    nc = len(names)
    idx = iou_index(ecfg["iou_thr"])
    thr = f"{ecfg['iou_thr']:.2f}".rstrip("0").rstrip(".")
    data_yaml = (ROOT / tcfg["data"]).resolve()
    check_dataset(data_yaml, names)

    env = environment_info()
    print("실행 환경:", env)
    if str(tcfg["device"]) != "cpu" and not env["cuda_available"]:
        sys.exit("[오류] CUDA를 쓸 수 없습니다. GPU PC에서 실행하거나 --device cpu 를 붙이세요.")

    boxes = {s: count_boxes(data_yaml, s, nc) for s in splits}
    rows = []
    for name, w in weight_list(args, tcfg):
        for s in splits:
            print(f"\n===== {name} / {SPLIT_DIRS[s]} 평가 =====")
            res = evaluate(name, w, s, tcfg, ecfg, data_yaml, idx)
            rows.append((name, s, res))
            missing = [names[i] for i in range(nc) if res["ap"][i] is None]
            if missing:
                print(f"[경고] {SPLIT_DIRS[s]}에 정답 박스가 없어 AP를 계산할 수 없는 클래스: {missing}")

    out = ROOT / ecfg["output"]
    out.mkdir(parents=True, exist_ok=True)

    # CSV: 한 줄 = 모델 x 분할
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "split", f"mAP@{thr}", "mAP@0.5:0.95", "target_met"]
                    + [f"AP@{thr}_{n}" for n in names] + [f"P_{n}" for n in names] + [f"R_{n}" for n in names])
        for name, s, res in rows:
            met = res["map"] is not None and res["map"] >= ecfg["target_map"]
            wr.writerow([name, SPLIT_DIRS[s], fmt(res["map"]), fmt(res["map50_95"]), met]
                        + [fmt(v) for v in res["ap"] + res["p"] + res["r"]])

    # 마크다운: 노션 붙여넣기용
    focus = [names.index(c) for c in ecfg["focus_classes"]]
    lines = [f"## 탐지 성능 (mAP@{thr}, 목표 {ecfg['target_map']:.2f} 이상)", ""]
    lines += ["| 모델 | 분할 | mAP@" + thr + " | 목표 달성 | " + " | ".join(f"AP {n}" for n in names)
              + " | " + " | ".join(f"Recall {names[i]}" for i in focus) + " |"]
    lines += ["| --- " * (4 + nc + len(focus)) + "|"]
    for name, s, res in rows:
        met = "O" if res["map"] is not None and res["map"] >= ecfg["target_map"] else "X"
        lines.append(f"| {name} | {SPLIT_DIRS[s]} | {fmt(res['map'])} | {met} | "
                     + " | ".join(fmt(v) for v in res["ap"]) + " | "
                     + " | ".join(fmt(res["r"][i]) for i in focus) + " |")
    lines += ["", "## 분할별 정답 박스 수", "", "| 분할 | " + " | ".join(names) + " |",
              "| --- " * (nc + 1) + "|"]
    for s in splits:
        lines.append(f"| {SPLIT_DIRS[s]} | " + " | ".join(str(n) for n in boxes[s]) + " |")
    lines += ["", "- 박스 수가 적은 클래스(예: valid no-helmet)는 AP가 크게 흔들릴 수 있으므로 valid와 test를 함께 봅니다.",
              "- Precision·Recall은 F1이 가장 높은 신뢰도 기준 값입니다(Ultralytics). 전체 값은 results.csv에 있습니다.",
              f"- 실행 환경: {env.get('gpu', 'CPU')}, torch {env['torch']}, ultralytics {env['ultralytics']}"]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\n저장: {out / 'results.csv'}, {out / 'results.md'}")


if __name__ == "__main__":
    main()
