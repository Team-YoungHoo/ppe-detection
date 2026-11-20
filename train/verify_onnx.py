"""TRN-06 ONNX 변환 결과 확인 스크립트.

같은 입력(letterbox 전처리)을 PyTorch(best.pt), ONNX Runtime, OpenCV DNN 에 넣고 출력이 같은지 비교합니다.
  - 원시 출력 비교: 박스 좌표(행 0~3)와 클래스 점수(행 4~)의 최대 차이
  - 탐지 결과 비교: 신뢰도 필터 + 클래스별 NMS 후 박스 수, 같은 클래스 박스끼리 IoU
속도는 재지 않습니다. 결과는 runs/export/verify_<provider>.md 에 저장됩니다(이슈에 붙여넣기용).

실행 PC:
  --provider cpu  : 노트북에서 실행해도 됩니다.
  --provider cuda : 실습 PC(GPU)에서 실행합니다. 실제 실행 엔진(ONNX Runtime CUDA)에서의 최종 확인입니다.

사용 예 (저장소 루트에서 실행):
    python train/verify_onnx.py --weights path/to/best.pt
    python train/verify_onnx.py --weights path/to/best.pt --images path/to/test/images --max-images 10
    python train/verify_onnx.py --weights path/to/best.pt --provider cuda
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="ONNX 변환 결과 확인 (TRN-06)")
    p.add_argument("--weights", required=True, help="변환에 쓴 best.pt 경로")
    p.add_argument("--onnx", default=str(ROOT / "models" / "best.onnx"), help="확인할 ONNX 경로")
    p.add_argument("--meta", default=str(ROOT / "models" / "model.yaml"), help="export_onnx.py 가 만든 model.yaml")
    p.add_argument("--images", nargs="+", default=[str(ROOT / "tests" / "bus.jpg")],
                   help="비교할 이미지 파일 또는 폴더 (기본 tests/bus.jpg)")
    p.add_argument("--max-images", type=int, default=10, help="폴더에서 쓸 최대 이미지 수 (기본 10)")
    p.add_argument("--provider", choices=["cpu", "cuda"], default="cpu", help="ONNX Runtime 실행 장치")
    p.add_argument("--conf", type=float, default=0.25, help="탐지 비교용 신뢰도 임계값")
    p.add_argument("--iou", type=float, default=0.45, help="탐지 비교용 NMS IoU 임계값")
    p.add_argument("--tol-box", type=float, default=0.5, help="허용 박스 좌표 차이 (입력 픽셀)")
    p.add_argument("--tol-score", type=float, default=1e-3, help="허용 점수 차이")
    return p.parse_args()


def load_meta(path):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"[오류] model.yaml 이 없습니다: {path}\n  먼저 python train/export_onnx.py 를 실행하세요.")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def image_list(items, limit):
    files = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)[:limit]
        elif p.is_file():
            files.append(p)
        else:
            print(f"[경고] 없는 경로라 건너뜁니다: {p}")
    if not files:
        sys.exit("[오류] 비교할 이미지가 없습니다.")
    return files


def letterbox(img, size, pad_value):
    """C++ 전처리(VIN-02)와 같은 규칙: 비율 유지 리사이즈 후 가운데 정렬, 남는 곳은 pad_value."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)


def make_blob(img, meta):
    pre = meta["preprocess"]
    lb = letterbox(img, meta["input"]["width"], pre["pad_value"])
    return cv2.dnn.blobFromImage(lb, pre["scale"], swapRB=pre["swap_rb"]).astype(np.float32)


# ---------- 세 가지 실행 엔진 ----------

def torch_runner(weights):
    import torch
    from ultralytics import YOLO

    net = YOLO(str(weights)).model.float().eval()
    try:
        net = net.fuse(verbose=False)       # 변환할 때처럼 Conv+BN 합치기
    except TypeError:
        net = net.fuse()

    def run(blob):
        with torch.no_grad():
            y = net(torch.from_numpy(blob))
        if isinstance(y, (list, tuple)):     # eval 모드 Detect 출력: (y, 중간값)
            y = y[0]
        return y.cpu().numpy()

    return run, f"PyTorch {torch.__version__} (CPU)"


def ort_runner(onnx_path, provider):
    import onnxruntime as ort

    if provider == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            sys.exit("[오류] 이 PC의 onnxruntime 에 CUDA가 없습니다. onnxruntime-gpu 를 설치한 실습 PC에서 실행하세요.\n"
                     f"  사용 가능: {ort.get_available_providers()}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    used = sess.get_providers()[0]
    if provider == "cuda" and used != "CUDAExecutionProvider":
        # CUDA 등록에 실패하면 ORT가 조용히 CPU로 돌기 때문에 직접 확인합니다(CUDA·cuDNN 버전 불일치 등).
        sys.exit(f"[오류] CUDA 실행 장치 등록에 실패해 {used} 로 실행됩니다. CUDA·cuDNN 버전을 확인하세요.")
    name = sess.get_inputs()[0].name

    def run(blob):
        return sess.run(None, {name: blob})[0]

    return run, f"ONNX Runtime {ort.__version__} ({used})"


def cv_runner(onnx_path):
    try:
        net = cv2.dnn.readNetFromONNX(str(onnx_path))
    except cv2.error as e:
        print(f"[경고] OpenCV DNN 이 ONNX 를 읽지 못했습니다 (opset 호환 문제일 수 있음): {e}")
        return None, f"OpenCV DNN {cv2.__version__} (읽기 실패)"

    def run(blob):
        net.setInput(blob)
        return net.forward()

    return run, f"OpenCV DNN {cv2.__version__} (CPU)"


# ---------- 비교 ----------

def decode(out, conf, iou):
    """{1, 4+nc, N} -> (박스 xyxy, 점수, 클래스). C++ 후처리(DET-03)와 같은 방식."""
    pred = out[0].T                              # (N, 4+nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(1)
    score = scores_all[np.arange(len(cls)), cls]
    keep = score >= conf
    pred, cls, score = pred[keep], cls[keep], score[keep]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xywh = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
    idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), score.tolist(), cls.tolist(), conf, iou) if len(cls) else []
    idx = np.array(idx, dtype=int).reshape(-1)
    xyxy = np.stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], 1)
    return xyxy[idx], score[idx], cls[idx]


def box_iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def compare_dets(ref, other):
    """기준(PyTorch) 탐지마다 같은 클래스 중 IoU가 가장 큰 박스를 찾아 최소 IoU를 구합니다."""
    rb, rs, rc = ref
    ob, os_, oc = other
    if len(rb) == 0:
        return len(ob) == 0, 1.0
    min_iou = 1.0
    for b, c in zip(rb, rc):
        cand = ob[oc == c]
        best = box_iou(b, cand).max() if len(cand) else 0.0
        min_iou = min(min_iou, float(best))
    return len(rb) == len(ob), min_iou


def main():
    args = parse_args()
    meta = load_meta(args.meta)
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        sys.exit(f"[오류] ONNX 파일이 없습니다: {onnx_path}")
    images = image_list(args.images, args.max_images)

    engines = {}
    engines["torch"] = torch_runner(args.weights)
    engines["ort"] = ort_runner(onnx_path, args.provider)
    engines["cv"] = cv_runner(onnx_path)

    pairs = [("ort", "ONNX Runtime"), ("cv", "OpenCV DNN")]
    rows = []                                    # (이미지, 엔진, 박스 차이, 점수 차이, 탐지 수 같음, 최소 IoU, 통과)
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[경고] 이미지를 읽지 못해 건너뜁니다: {img_path}")
            continue
        blob = make_blob(img, meta)
        ref = engines["torch"][0](blob)
        ref_det = decode(ref, args.conf, args.iou)
        for key, label in pairs:
            run = engines[key][0]
            if run is None:
                rows.append((img_path.name, label, None, None, False, 0.0, False))
                continue
            out = run(blob)
            if out.shape != ref.shape:
                sys.exit(f"[오류] {label} 출력 형태 {out.shape} 가 PyTorch {ref.shape} 와 다릅니다.")
            d_box = float(np.abs(out[0, :4] - ref[0, :4]).max())
            d_score = float(np.abs(out[0, 4:] - ref[0, 4:]).max())
            same_n, min_iou = compare_dets(ref_det, decode(out, args.conf, args.iou))
            ok = d_box <= args.tol_box and d_score <= args.tol_score and same_n
            rows.append((img_path.name, label, d_box, d_score, same_n, min_iou, ok))
            print(f"{img_path.name:30s} {label:12s} 박스차 {d_box:.4f}  점수차 {d_score:.6f}  "
                  f"탐지수 {'같음' if same_n else '다름'} (PyTorch {len(ref_det[0])}개)  최소IoU {min_iou:.4f}  "
                  f"{'통과' if ok else '실패'}")

    all_ok = all(r[6] for r in rows)
    fmt = lambda v, n: "N/A" if v is None else f"{v:.{n}f}"
    lines = [
        f"## ONNX 변환 확인 ({args.provider.upper()})", "",
        f"- 모델: {meta.get('model_name')} / 입력 {meta['input']['shape']} / 출력 {meta['output']['shape']} / opset {meta.get('opset')}",
        f"- 기준: {engines['torch'][1]}",
        f"- 비교: {engines['ort'][1]}, {engines['cv'][1]}",
        f"- 허용 차이: 박스 {args.tol_box} px, 점수 {args.tol_score} / 탐지 비교 conf {args.conf}, NMS IoU {args.iou}",
        f"- 이미지 {len(images)}장, 실행 {datetime.now().isoformat(timespec='seconds')}", "",
        "| 이미지 | 엔진 | 박스 최대 차이(px) | 점수 최대 차이 | 탐지 수 같음 | 최소 IoU | 결과 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, label, d_box, d_score, same_n, min_iou, ok in rows:
        lines.append(f"| {name} | {label} | {fmt(d_box, 4)} | {fmt(d_score, 6)} | {'O' if same_n else 'X'} | "
                     f"{min_iou:.4f} | {'통과' if ok else '실패'} |")
    lines += ["", f"**전체 결과: {'통과' if all_ok else '실패'}**"]

    out_dir = ROOT / "runs" / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"verify_{args.provider}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n전체 결과: {'통과' if all_ok else '실패'}  (저장: {report})")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
