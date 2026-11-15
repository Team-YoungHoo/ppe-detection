"""TRN-05 모델 크기별 성능 비교 스크립트.

모델마다 정확도(evaluate.py 결과)와 속도(직접 측정), 파라미터 수를 한 표로 모으고,
"mAP@0.5 >= 목표" 와 "FPS >= 목표" 를 모두 만족하는 모델 중 가장 작은 것을 잠정 선정합니다.
결과: runs/compare/compare.csv, runs/compare/compare.md (노션에 붙여넣기용)

순서 (저장소 루트에서 실행):
    python train/train.py
    python train/evaluate.py
    python train/scripts/compare_models.py

주의: TRN-05의 원래 속도 기준은 C++ 전체 파이프라인 FPS(EVL-02)입니다. C++ 파이프라인이 아직 없으므로
      여기서는 Python(PyTorch) 처리 시간(전처리+추론+후처리)으로 잠정 판단합니다. 최종 선정은 EVL-02 측정 후 확정합니다.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # train/ 폴더의 train.py, evaluate.py 사용
from evaluate import SPLIT_DIRS  # noqa: E402
from train import DEFAULT_CONFIG, ROOT, environment_info  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="모델 크기별 정확도·속도 비교 (TRN-05)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="설정 파일 경로")
    p.add_argument("--models", nargs="+", help="비교할 모델 (기본: 설정 파일의 train.models)")
    p.add_argument("--name-suffix", default="", help="학습 때 쓴 실험 이름 접미사 (예: _smoke)")
    p.add_argument("--device", help="설정값 대신 쓸 장치 (예: 0, cpu)")
    return p.parse_args()


def load_configs(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("train", "eval", "compare"):
        if not cfg or key not in cfg:
            sys.exit(f"[오류] {path} 에 {key} 설정이 없습니다.")
    return cfg["train"], cfg["eval"], cfg["compare"]


def read_eval_results(ecfg, split):
    """evaluate.py가 만든 results.csv 에서 {모델 이름: mAP} 를 읽습니다."""
    path = ROOT / ecfg["output"] / "results.csv"
    if not path.is_file():
        sys.exit(f"[오류] {path} 가 없습니다. train/evaluate.py 를 먼저 실행하세요.")
    thr = f"{ecfg['iou_thr']:.2f}".rstrip("0").rstrip(".")
    col = f"mAP@{thr}"
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"] == SPLIT_DIRS[split] and row[col] != "N/A":
                out[row["model"]] = float(row[col])
    return out, col


def test_images(data_yaml, split):
    with open(data_yaml, encoding="utf-8") as f:
        rel = yaml.safe_load(f)[split]
    img_dir = data_yaml.parent / rel
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        sys.exit(f"[오류] 이미지가 없습니다: {img_dir}")
    return imgs


def measure_speed(model, imgs, tcfg, ccfg):
    """이미지 1장씩(batch 1) 처리 시간을 잽니다. 워밍업을 버리고 repeats번 반복한 평균(ms/이미지)을 돌려줍니다."""
    def run(paths):
        pre, inf, post = [], [], []
        for r in model.predict(source=[str(p) for p in paths], imgsz=tcfg["imgsz"], device=tcfg["device"],
                               batch=1, stream=True, verbose=False):
            pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
        return statistics.mean(pre), statistics.mean(inf), statistics.mean(post)

    run(imgs[: ccfg["warmup"]])                          # 워밍업 (결과 버림)
    passes = [run(imgs) for _ in range(ccfg["repeats"])]
    pre, inf, post = (statistics.mean(x) for x in zip(*passes))
    return {"pre_ms": pre, "inf_ms": inf, "post_ms": post, "total_ms": pre + inf + post,
            "inf_ms_runs": [round(p[1], 2) for p in passes]}


def main():
    args = parse_args()
    tcfg, ecfg, ccfg = load_configs(args.config)
    if args.device is not None:
        tcfg["device"] = args.device

    env = environment_info()
    print("실행 환경:", env)
    if str(tcfg["device"]) != "cpu" and not env["cuda_available"]:
        sys.exit("[오류] CUDA를 쓸 수 없습니다. 속도는 실제 실행할 GPU PC에서 재야 합니다 (동작 확인만 할 때는 --device cpu).")

    maps, map_col = read_eval_results(ecfg, ccfg["select_split"])
    data_yaml = (ROOT / tcfg["data"]).resolve()
    imgs = test_images(data_yaml, ccfg["speed_split"])

    from ultralytics import YOLO

    rows = []
    for m in args.models or tcfg["models"]:
        name = m + args.name_suffix
        w = ROOT / tcfg["project"] / name / "weights" / "best.pt"
        if not w.is_file():
            print(f"[경고] 가중치가 없어 건너뜁니다: {w}")
            continue
        if name not in maps:
            print(f"[경고] {name}의 {SPLIT_DIRS[ccfg['select_split']]} mAP가 results.csv에 없어 건너뜁니다. evaluate.py를 다시 실행하세요.")
            continue
        print(f"\n===== {name} 속도 측정 ({len(imgs)}장 x {ccfg['repeats']}회) =====")
        model = YOLO(str(w))
        params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
        sp = measure_speed(model, imgs, tcfg, ccfg)
        fps = 1000 / sp["total_ms"]
        rows.append({"model": name, "params_m": params_m, "map": maps[name], "fps": fps, **sp,
                     "map_ok": maps[name] >= ecfg["target_map"], "fps_ok": fps >= ccfg["target_fps"]})

    if not rows:
        sys.exit("[오류] 비교할 모델이 없습니다.")
    rows.sort(key=lambda r: r["params_m"])                 # 작은 모델부터
    ok = [r for r in rows if r["map_ok"] and r["fps_ok"]]
    chosen = ok[0] if ok else None

    out = ROOT / ccfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    split_name = SPLIT_DIRS[ccfg["select_split"]]
    with open(out / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "params_M", f"{map_col}_{split_name}", "pre_ms", "inference_ms", "post_ms",
                     "total_ms", "fps", "inference_ms_each_run", "map_ok", "fps_ok", "selected"])
        for r in rows:
            wr.writerow([r["model"], f"{r['params_m']:.1f}", f"{r['map']:.3f}", f"{r['pre_ms']:.2f}",
                         f"{r['inf_ms']:.2f}", f"{r['post_ms']:.2f}", f"{r['total_ms']:.2f}", f"{r['fps']:.1f}",
                         r["inf_ms_runs"], r["map_ok"], r["fps_ok"], r is chosen])

    lines = [f"## 모델 크기별 비교 (선정 기준: {split_name} {map_col} ≥ {ecfg['target_map']:.2f}, "
             f"FPS ≥ {ccfg['target_fps']} 중 가장 작은 모델)", "",
             f"| 모델 | 파라미터(M) | {map_col} ({split_name}) | 추론(ms) | 전체 처리(ms) | FPS | 정확도 목표 | 속도 목표 | 선정 |",
             "| --- " * 9 + "|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['params_m']:.1f} | {r['map']:.3f} | {r['inf_ms']:.2f} | {r['total_ms']:.2f} | "
                     f"{r['fps']:.1f} | {'O' if r['map_ok'] else 'X'} | {'O' if r['fps_ok'] else 'X'} | "
                     f"{'**잠정 선정**' if r is chosen else ''} |")
    lines.append("")
    if chosen:
        lines.append(f"- 잠정 선정: **{chosen['model']}**. C++ 전체 파이프라인 FPS(EVL-02) 측정 후 확정합니다.")
    else:
        lines.append("- 두 목표를 모두 만족하는 모델이 없습니다. 이 표로 팀에 보고하고 목표 조정 또는 입력 크기 축소를 논의합니다 (TRN-05 예외 처리).")
    lines += [f"- 속도: {ccfg['speed_split']} 이미지 {len(imgs)}장, batch 1, imgsz {tcfg['imgsz']}, "
              f"워밍업 {ccfg['warmup']}장 제외, {ccfg['repeats']}회 평균. Python(PyTorch) 기준이며 ONNX Runtime·C++ 값과 다를 수 있습니다.",
              "- 전체 처리(ms) = 전처리 + 추론 + 후처리(NMS). 영상 읽기·추적·판단·화면 출력은 포함하지 않습니다.",
              f"- 실행 환경: {env.get('gpu', 'CPU')}, torch {env['torch']}, ultralytics {env['ultralytics']}"]
    (out / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\n저장: {out / 'compare.csv'}, {out / 'compare.md'}")


if __name__ == "__main__":
    main()
