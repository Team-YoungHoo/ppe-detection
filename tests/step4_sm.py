# step4_sm.py

import numpy as np

for name in ["yolov8s", "yolov8m", "yolo11s", "yolo11m"]:
    try:
        a = np.fromfile(f"{name}_ort.bin", dtype=np.float32)
        b = np.fromfile(f"{name}_cv.bin", dtype=np.float32)
        if a.size != b.size:
            print(name, "| 출력 크기 다름:", a.size, b.size)
        else:
            print(name, "| 최대 오차:", float(np.abs(a - b).max()))
    except FileNotFoundError:
        print(name, "| 파일 없음 (로드나 실행 실패)")