// DET-07 모델 계약 파일(models/model.yaml) 읽기
//
// 학습·변환(Python)과 실행(C++)이 같은 규칙을 쓰도록 export_onnx.py가 만든 값을 읽고 검사함

#pragma once

#include <string>
#include <vector>

struct ModelMeta {
    std::string model_name;          // 모델 이름
    int imgsz = 0;                   // 모델 입력 크기 (input.width = input.height)
    int nc = 0;                      // 클래스 수
    std::vector<std::string> names;  // 클래스 이름 (순서 = class_id)
    std::vector<int> output_shape;   // 출력 형태 {1, 4 + nc, N}
    int pad_value = 0;               // Letterbox 패딩 값
    double scale = 0.0;              // 픽셀 정규화 배율
    bool swap_rb = false;            // BGR -> RGB 변환 여부

    // 파일을 읽고 값을 검사함. 실패하면 false와 이유를 반환하고 호출한 쪽에서 종료 코드 3으로 끝냄
    static bool load(const std::string& path, ModelMeta& out, std::string& error);
};
