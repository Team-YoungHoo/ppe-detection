import time
import numpy as np
import onnxruntime as ort

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # DirectML이면 "DmlExecutionProvider"
models = ["yolov8s", "yolov8m", "yolo11s", "yolo11m"]

print("사용 가능:", ort.get_available_providers())
blob = np.fromfile("input.bin", dtype=np.float32).reshape(1, 3, 640, 640)

for name in models:
    print("==", name)
    try:
        sess = ort.InferenceSession(f"{name}.onnx", providers=PROVIDERS)
        print("  실제 사용:", sess.get_providers())   # 첫 번째가 CUDA여야 GPU로 실행되는 것
        inp = sess.get_inputs()[0].name

        for _ in range(3):                             # GPU는 첫 실행이 느려서 워밍업 여러 번
            out = sess.run(None, {inp: blob})
        t = time.time()
        for _ in range(20):
            out = sess.run(None, {inp: blob})
        ms = (time.time() - t) / 20 * 1000
        print("  출력 shape:", out[0].shape)
        print(f"  GPU 평균 시간: {ms:.1f} ms")

        cpu = np.fromfile(f"{name}_ort.bin", dtype=np.float32)   # 기존 CPU 결과와 비교
        print("  CPU 대비 최대 오차:", float(np.abs(out[0].astype(np.float32).ravel() - cpu).max()))
    except Exception as e:
        print("  실패:", e)