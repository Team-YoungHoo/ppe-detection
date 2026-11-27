"""TRN-06 학습된 모델 ONNX 변환 스크립트.

학습 결과(best.pt)를 C++ 추론 모듈이 읽을 ONNX 파일로 바꾸고, 함께 쓸 정보 파일을 만듭니다(DET-07).
결과 (기본 위치 models/):
    best.onnx    ONNX 모델 (입력 {1,3,imgsz,imgsz}, 출력 {1,4+nc,N}, NMS 미포함)
    classes.txt  클래스 이름, 한 줄에 하나 (줄 번호 = 클래스 번호)
    model.yaml   입력 크기, 출력 형태, 전처리 값, 학습 설정, 변환 환경, 해시

GPU가 필요 없으므로 노트북에서 실행해도 됩니다. 변환 결과가 맞는지는 train/verify_onnx.py 로 확인합니다.

사용 예 (저장소 루트에서 실행):
    python train/export_onnx.py --weights path/to/best.pt
    python train/export_onnx.py --weights path/to/best.pt --sha256 5247e9d9...  # 받은 파일이 맞는지 확인
    python train/export_onnx.py --weights path/to/best.pt --imgsz 960           # 960 모델로 바뀔 경우
"""

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]          # 저장소 루트

# 클래스 순서는 고정입니다(configs/default.yaml train.names, C++ 설정과 같아야 함). 순서 변경 금지.
CLASS_NAMES = ["person", "helmet", "vest", "no-helmet", "no-vest"]

# C++ 전처리 규칙 (VIN-02/03). C++ 쪽은 이 값을 model.yaml 에서 읽어 맞춥니다.
PREPROCESS = {
    "resize": "letterbox",      # 비율 유지 축소/확대 후 가운데 정렬, 남는 곳은 채움
    "pad_value": 114,           # Ultralytics 학습 시 채움 값과 같음
    "scale": 1.0 / 255.0,
    "swap_rb": True,            # OpenCV BGR -> 모델 입력 RGB
    "layout": "NCHW",
    "dtype": "float32",
}

# 학습 기록에서 model.yaml 에 옮겨 적을 항목 (팀원 공유 사항의 증강 값 포함)
TRAIN_KEYS = ["model", "data", "imgsz", "epochs", "batch", "seed",
              "mixup", "scale", "degrees", "fliplr", "mosaic", "hsv_h", "hsv_s", "hsv_v"]


class QuotedStr(str):
    """model.yaml 에 큰따옴표로 저장할 문자열입니다."""


class MetaDumper(yaml.SafeDumper):
    """OpenCV cv::FileStorage 가 읽을 수 있도록 목록을 들여쓰기해서 저장합니다."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


MetaDumper.add_representer(
    QuotedStr, lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:str", str(value), style='"')
)


def quote_strings(value):
    """숫자로 시작하는 해시 등을 OpenCV 가 숫자로 읽지 않도록 문자열 값에 따옴표를 붙입니다."""
    if isinstance(value, dict):
        return {k: quote_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [quote_strings(v) for v in value]
    if isinstance(value, str):
        return QuotedStr(value)
    return value


def write_model_yaml(path, meta):
    """PyYAML 과 OpenCV cv::FileStorage 가 모두 읽는 형식으로 model.yaml 을 저장합니다."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("%YAML 1.1\n---\n")     # cv::FileStorage 는 첫 줄에 이 표시가 있어야 YAML 로 인식함
        f.write("# TRN-06 / DET-07: export_onnx.py 가 만든 파일입니다. 직접 고치지 마세요.\n")
        yaml.dump(quote_strings(meta), f, Dumper=MetaDumper, allow_unicode=True, sort_keys=False,
                  default_flow_style=False, width=100000)


def parse_args():
    p = argparse.ArgumentParser(description="학습된 모델 ONNX 변환 (TRN-06, DET-07)")
    p.add_argument("--weights", required=True, help="변환할 best.pt 경로")
    p.add_argument("--sha256", help="받은 best.pt 의 SHA-256 (주면 다를 때 중단)")
    p.add_argument("--imgsz", type=int, default=640, help="입력 크기 (학습 imgsz와 같아야 함, 기본 640)")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset (OpenCV DNN 호환용, 기본 12)")
    p.add_argument("--no-simplify", action="store_true", help="ONNX 그래프 단순화(onnxslim) 끄기")
    p.add_argument("--out-dir", default=str(ROOT / "models"), help="결과 폴더 (기본 models/)")
    p.add_argument("--name", default="best", help="결과 파일 이름 (기본 best -> best.onnx)")
    p.add_argument("--model-name", help="model.yaml 에 적을 모델 이름 (예: yolo11s_v3_aug)")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def environment_info():
    import cv2
    import onnx
    import torch
    import ultralytics

    info = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": str(torch.__version__),
        "ultralytics": str(ultralytics.__version__),
        "onnx": str(onnx.__version__),
        "opencv": str(cv2.__version__),
    }
    try:
        import onnxruntime
        info["onnxruntime"] = str(onnxruntime.__version__)
    except ImportError:
        info["onnxruntime"] = "not installed"
    return info


def check_classes(model):
    names = model.names
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if list(names) != CLASS_NAMES:
        sys.exit(f"[오류] 모델의 클래스 순서가 고정 순서와 다릅니다.\n  모델: {names}\n  고정: {CLASS_NAMES}")
    return list(names)


def train_record(model):
    """best.pt 안에 저장된 학습 설정을 꺼냅니다. 없는 항목은 건너뜁니다."""
    ckpt = getattr(model, "ckpt", None) or {}
    args = ckpt.get("train_args") or {}
    rec = {k: args[k] for k in TRAIN_KEYS if k in args}
    if ckpt.get("date"):
        rec["date"] = str(ckpt["date"])[:19]   # 학습 날짜 (best.pt 저장 시각, 초 단위)
    if "data" in rec:
        # 학습 PC의 전체 경로 대신 데이터셋 폴더 이름만 남깁니다 (PC마다 경로가 다르고, 사용자 이름이 들어 있음)
        parts = str(rec["data"]).replace("\\", "/").rstrip("/").split("/")
        rec["data"] = parts[-2] if len(parts) >= 2 else parts[-1]
    return rec


def check_onnx(path, imgsz, nc):
    """저장된 ONNX의 입출력 형태가 C++ 추론 모듈이 기대하는 형태인지 확인합니다."""
    import onnx

    m = onnx.load(str(path))
    onnx.checker.check_model(m)

    def dims(v):
        return [d.dim_value if d.dim_value > 0 else d.dim_param for d in v.type.tensor_type.shape.dim]

    if len(m.graph.input) != 1 or len(m.graph.output) != 1:
        sys.exit(f"[오류] 입력/출력이 1개씩이어야 합니다 (입력 {len(m.graph.input)}, 출력 {len(m.graph.output)})")
    inp, out = m.graph.input[0], m.graph.output[0]
    in_shape, out_shape = dims(inp), dims(out)

    if in_shape != [1, 3, imgsz, imgsz]:
        sys.exit(f"[오류] 입력 형태가 다릅니다: {in_shape} (기대 [1, 3, {imgsz}, {imgsz}])")
    # YOLOv8/11/26(one-to-many): {1, 4+nc, N}. 4 = 박스(cx, cy, w, h), 나머지 = 클래스별 점수. objectness 없음
    if len(out_shape) != 3 or out_shape[0] != 1 or out_shape[1] != 4 + nc:
        hint = ""
        if len(out_shape) == 3 and out_shape[2] == 6:
            hint = ("\n  NMS-free(end-to-end) 출력 {1, max_det, 6} 으로 보입니다. "
                    "export 인자가 nms=None 인지, ultralytics 버전이 nms 인자를 지원하는지 확인하세요.")
        sys.exit(f"[오류] 출력 형태가 다릅니다: {out_shape} (기대 [1, {4 + nc}, N]){hint}")
    # 박스 후보 수: stride 8, 16, 32 격자 칸 수의 합 (640 -> 8400, 960 -> 18900)
    expected_n = sum((imgsz // s) ** 2 for s in (8, 16, 32))
    if out_shape[2] != expected_n:
        sys.exit(f"[오류] 박스 후보 수가 다릅니다: {out_shape[2]} (기대 {expected_n})")

    return {"name": inp.name, "shape": in_shape}, {"name": out.name, "shape": out_shape}, m.opset_import[0].version


def main():
    args = parse_args()
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        sys.exit(f"[오류] 가중치 파일을 찾을 수 없습니다: {weights}")

    # 1. 받은 파일이 맞는지 확인
    pt_hash = sha256(weights)
    print(f"best.pt SHA-256: {pt_hash}")
    if args.sha256 and args.sha256.lower() != pt_hash:
        sys.exit(f"[오류] SHA-256 이 다릅니다. 받은 파일을 다시 확인하세요.\n  기대: {args.sha256.lower()}")

    from ultralytics import YOLO

    # 2. 모델 확인: 클래스 순서, 학습 imgsz
    model = YOLO(str(weights))
    names = check_classes(model)
    train_info = train_record(model)
    train_imgsz = train_info.get("imgsz")
    if train_imgsz is not None and int(train_imgsz) != args.imgsz:
        print(f"[경고] 학습 imgsz({train_imgsz})와 변환 imgsz({args.imgsz})가 다릅니다. 의도한 것이 맞는지 확인하세요.")

    env = environment_info()
    print("변환 환경:", env)

    # 3. ONNX 변환. 입력 크기 고정(dynamic=False), FP32, NMS는 C++에서 하므로 넣지 않음
    #    nms=None: NMS를 넣지 않은 one-to-many 출력 {1, 4+nc, N}. YOLOv8/11/26 모두 같은 형태가 나옵니다.
    #    nms=False는 쓰지 않습니다. YOLO26처럼 one-to-one 헤드가 있는 모델에서는 NMS-free 헤드를 골라
    #    출력이 {1, 300, 6}(x1, y1, x2, y2, 점수, 클래스)으로 바뀌어 C++ 후처리(DET-02~04)와 맞지 않습니다.
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=False,
        nms=None,
        batch=1,
        device="cpu",
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{args.name}.onnx"
    if Path(exported).resolve() != onnx_path:
        shutil.move(str(exported), str(onnx_path))   # best.pt 옆에 생긴 파일을 models/ 로 옮김

    # 4. 입출력 형태 확인
    inp, out, opset = check_onnx(onnx_path, args.imgsz, len(names))

    # 5. classes.txt, model.yaml 작성
    (out_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    meta = {
        "model_name": args.model_name or weights.parents[1].name,
        "onnx_file": onnx_path.name,
        "input": {**inp, "width": args.imgsz, "height": args.imgsz},
        "output": {
            **out,
            "format": "[1, 4+nc, N], 행 0~3 = cx, cy, w, h (입력 이미지 픽셀), 행 4~ = 클래스별 점수 (objectness 없음)",
            "nms_included": False,
        },
        "nc": len(names),
        "names": names,
        "preprocess": PREPROCESS,
        "opset": opset,
        "train": train_info,
        "source_weights": {"file": weights.name, "sha256": pt_hash},
        "onnx_sha256": sha256(onnx_path),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment": env,
    }
    write_model_yaml(out_dir / "model.yaml", meta)

    print("\n===== 변환 완료 =====")
    print(f"ONNX       : {onnx_path}")
    print(f"입력       : {inp['name']} {inp['shape']}")
    print(f"출력       : {out['name']} {out['shape']}")
    print(f"opset      : {opset}")
    print(f"학습 설정  : {train_info}")
    print(f"정보 파일  : {out_dir / 'classes.txt'}, {out_dir / 'model.yaml'}")
    print("다음 단계  : python train/verify_onnx.py 로 PyTorch / ONNX Runtime / OpenCV DNN 출력을 비교하세요.")


if __name__ == "__main__":
    main()
