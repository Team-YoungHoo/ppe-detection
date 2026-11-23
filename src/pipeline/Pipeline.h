// SYS-05 파이프라인 뼈대 (메인 루프)
//
// 순서: 영상 읽기 -> 전처리 -> (탐지·추적·판단·경고 자리) -> 화면 출력
// 빈 자리는 해당 기능의 이슈에서 채움

#pragma once

#include <string>

struct PipelineOptions {
    std::string input;       // 영상 파일 경로 (input.path)
    int model_size = 640;    // 모델 입력 크기 (Config 준비 전 임시 기본값)
    bool display = true;     // false이면 창 없이 처리만 함
};

class Pipeline {
public:
    explicit Pipeline(const PipelineOptions& options);

    // 메인 루프를 실행하고 종료 코드를 반환함 (0 = 정상, 2 = 입력 영상 오류)
    int run();

    // 마지막 run()에서 처리한 프레임 수
    int processedFrames() const { return processed_; }

private:
    PipelineOptions options_;
    int processed_ = 0;
};
