# step2_sm.py

import time
import numpy as np
import cv2
import onnx
import onnxruntime as ort

models = ["yolov8s", "yolov8m", "yolo11s", "yolo11m"]

img = cv2.imread("bus.jpg")
blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False).astype(np.float32)
blob.tofile("input.bin")
print("입력 shape:", blob.shape)

for name in models:
    print("==", name)
    try:
        ops = {n.op_type for n in onnx.load(f"{name}.onnx").graph.node}
        print("  TopK 포함:", "TopK" in ops)

        sess = ort.InferenceSession(f"{name}.onnx", providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name

        out = sess.run(None, {inp: blob})  # 워밍업
        t = time.time()
        for _ in range(5):
            out = sess.run(None, {inp: blob})
        ms = (time.time() - t) / 5 * 1000

        print("  출력 shape:", [o.shape for o in out])
        print(f"  CPU 평균 시간: {ms:.1f} ms")
        out[0].astype(np.float32).tofile(f"{name}_ort.bin")
    except Exception as e:
        print("  실패:", e)