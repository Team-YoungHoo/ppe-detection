"""VIN-03 검증: C++ 전처리(Letterbox + blob)와 Python(Ultralytics) 전처리의 입력 텐서 비교.

C++ test_letterbox --dump 가 저장한 blob 파일과, 같은 이미지를 Ultralytics LetterBox로
전처리한 텐서를 비교합니다. 상위 10개 값과 최대 차이를 출력합니다(기능 명세 VIN-03 검증 방법).

사용 예 (저장소 루트에서 실행):
    out\\build\\my-release\\test_letterbox.exe --dump tests\\bus.jpg runs\\blob_cpp.bin
    python tests/check_blob.py --image tests/bus.jpg --blob runs/blob_cpp.bin
"""

import argparse
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="C++ / Python 입력 텐서 비교 (VIN-03)")
    p.add_argument("--image", required=True, help="C++에 넣은 것과 같은 이미지")
    p.add_argument("--blob", required=True, help="test_letterbox --dump 결과 파일")
    p.add_argument("--imgsz", type=int, default=640, help="모델 입력 크기 (model.yaml input.width)")
    p.add_argument("--tol", type=float, default=1e-6, help="허용 차이")
    return p.parse_args()


def python_blob(img, size):
    """학습·변환 때와 같은 Ultralytics 전처리: LetterBox(114) -> RGB -> /255 -> NCHW."""
    from ultralytics.data.augment import LetterBox

    lb = LetterBox(new_shape=(size, size), auto=False, scale_fill=False, scaleup=True, center=True)
    x = lb(image=img)                                   # S x S BGR, 패딩 114
    x = x[..., ::-1].transpose(2, 0, 1)[None]           # BGR -> RGB, HWC -> NCHW
    return np.ascontiguousarray(x, dtype=np.float32) / 255.0


def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"[오류] 이미지를 읽을 수 없습니다: {args.image}")

    ref = python_blob(img, args.imgsz)
    cpp = np.fromfile(args.blob, dtype=np.float32)
    if cpp.size != ref.size:
        sys.exit(f"[오류] 텐서 크기가 다릅니다: C++ {cpp.size}, Python {ref.size} ({ref.shape})")
    cpp = cpp.reshape(ref.shape)

    diff = np.abs(cpp - ref)
    print(f"이미지: {args.image} ({img.shape[1]}x{img.shape[0]}) -> 텐서 {list(ref.shape)}")
    print(f"최대 차이: {diff.max():.8f}, 다른 값 개수: {int((diff > args.tol).sum())} / {diff.size}")

    # 상위 10개 값 비교 (값이 큰 순서, 같은 위치)
    idx = np.argsort(ref.ravel())[::-1][:10]
    print("\n위치(c, y, x)        Python      C++")
    for i in idx:
        c, y, x = np.unravel_index(i, ref.shape)[1:]
        print(f"({c}, {y:3d}, {x:3d})        {ref.ravel()[i]:.6f}    {cpp.ravel()[i]:.6f}")

    ok = diff.max() <= args.tol
    print(f"\n결과: {'일치' if ok else '불일치'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
