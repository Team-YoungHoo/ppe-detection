"""DET-02 검증: C++ 탐지 결과와 Python(Ultralytics) 탐지 결과를 비교합니다.

test_detector_compare 가 저장한 CSV와, 같은 이미지를 best.pt 로 추론한 결과를 비교합니다.
기준: 클래스와 탐지 수 일치, 박스 오차 2픽셀 이내, 점수 오차 0.01 이내 (기능 명세 DET-02).

사용 예 (저장소 루트에서 실행):
    python tests/check_detections.py --weights models/best.pt --meta models/model.yaml \\
        --images <이미지 폴더> --csv runs/det_cpp.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="C++ / Python 탐지 결과 비교 (DET-02)")
    p.add_argument("--weights", required=True, help="변환에 쓴 best.pt")
    p.add_argument("--meta", required=True, help="model.yaml")
    p.add_argument("--images", required=True, help="C++에 넣은 것과 같은 이미지 폴더 또는 파일")
    p.add_argument("--csv", required=True, help="test_detector_compare 결과 파일")
    p.add_argument("--max-images", type=int, default=10, help="폴더에서 쓸 이미지 수 (기본 10)")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 값")
    p.add_argument("--tol-box", type=float, default=2.0, help="허용 박스 좌표 차이 (원본 픽셀)")
    p.add_argument("--tol-score", type=float, default=0.01, help="허용 점수 차이")
    return p.parse_args()


def image_list(path, limit):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        sys.exit(f"[오류] 이미지 경로가 없습니다: {path}")
    return sorted((f for f in path.iterdir() if f.suffix.lower() in IMG_EXTS), key=lambda f: f.name)[:limit]


def read_cpp(path):
    """이미지 이름별 탐지 결과를 읽습니다. class_id -1은 탐지가 0개인 이미지입니다."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            items = rows.setdefault(r["image"], [])
            if int(r["class_id"]) >= 0:
                items.append((int(r["class_id"]), float(r["score"]),
                              np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])))
    return rows


def python_detect(net, img, size, conf, iou):
    """Ultralytics 전처리와 NMS로 탐지하고 박스를 원본 좌표로 되돌립니다."""
    x = LetterBox(new_shape=(size, size), auto=False, scale_fill=False, scaleup=True, center=True)(image=img)
    x = x[..., ::-1].transpose(2, 0, 1)[None]
    blob = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32) / 255.0)
    with torch.no_grad():
        y = net(blob)
    y = y[0] if isinstance(y, (list, tuple)) else y
    det = non_max_suppression(y, conf_thres=conf, iou_thres=iou)[0]
    det[:, :4] = ops.scale_boxes((size, size), det[:, :4], img.shape[:2])
    return [(int(d[5]), float(d[4]), d[:4].numpy().astype(np.float64)) for d in det]


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(py, cpp):
    """같은 클래스 안에서 IoU가 가장 큰 박스끼리 짝지어 최대 차이를 구합니다."""
    if len(py) != len(cpp):
        return False, float("inf"), float("inf")
    left = list(cpp)
    max_box, max_score = 0.0, 0.0
    for cls, score, box in py:
        cand = [c for c in left if c[0] == cls]
        if not cand:
            return False, float("inf"), float("inf")
        best = max(cand, key=lambda c: iou(box, c[2]))
        left = [c for c in left if c is not best]
        max_box = max(max_box, float(np.abs(box - best[2]).max()))
        max_score = max(max_score, abs(score - best[1]))
    return True, max_box, max_score


def main():
    args = parse_args()
    with open(args.meta, encoding="utf-8") as f:
        size = yaml.safe_load(f)["input"]["width"]
    net = YOLO(args.weights).model.float().eval().fuse(verbose=False)
    cpp_rows = read_cpp(args.csv)

    images = image_list(args.images, args.max_images)
    print(f"{'이미지':<28}{'Python':>8}{'C++':>6}{'박스 차이(px)':>15}{'점수 차이':>12}  결과")
    all_ok = bool(images)
    for path in images:
        if path.name not in cpp_rows:
            print(f"{path.name[:26]:<28} CSV에 없음  실패")
            all_ok = False
            continue
        img = cv2.imread(str(path))
        py = python_detect(net, img, size, args.conf, args.iou)
        cpp = cpp_rows[path.name]
        same, box, score = compare(py, cpp)
        ok = same and box <= args.tol_box and score <= args.tol_score
        all_ok = all_ok and ok
        print(f"{path.name[:26]:<28}{len(py):>8}{len(cpp):>6}{box:>15.4f}{score:>12.6f}  {'통과' if ok else '실패'}")

    print(f"\n허용 차이: 박스 {args.tol_box} px, 점수 {args.tol_score}")
    print(f"전체 결과: {'통과' if all_ok else '실패'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
