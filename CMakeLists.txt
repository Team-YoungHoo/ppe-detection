cmake_minimum_required(VERSION 3.21)
project(safety_app CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

# 모듈 코드 (main.cpp를 뺀 나머지)
# safety_app과 테스트가 같이 씀
add_library(ppe_core STATIC
  src/config/ModelMeta.cpp
  src/detect/Detector.cpp
  src/detect/OpenCvDnnDetector.cpp
  src/io/VideoSource.cpp
  src/pipeline/Pipeline.cpp
  src/preprocess/Letterbox.cpp
  src/ui/DebugDraw.cpp
  src/ui/MonitorWindow.cpp
)
target_include_directories(ppe_core PUBLIC src)
target_link_libraries(ppe_core PUBLIC ${OpenCV_LIBS})

add_executable(safety_app src/main.cpp)
target_link_libraries(safety_app PRIVATE ppe_core)

# 단위 테스트 (ctest 또는 exe 직접 실행)
enable_testing()
add_executable(test_video_source tests/test_video_source.cpp)
target_link_libraries(test_video_source PRIVATE ppe_core)
add_test(NAME video_source COMMAND test_video_source)
add_executable(test_letterbox tests/test_letterbox.cpp)
target_link_libraries(test_letterbox PRIVATE ppe_core)
add_test(NAME letterbox COMMAND test_letterbox)
add_executable(test_monitor_window tests/test_monitor_window.cpp)
target_link_libraries(test_monitor_window PRIVATE ppe_core)
add_test(NAME monitor_window COMMAND test_monitor_window)
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE ppe_core)
add_test(NAME pipeline COMMAND test_pipeline)
add_executable(test_model_meta tests/test_model_meta.cpp)
target_link_libraries(test_model_meta PRIVATE ppe_core)
add_test(NAME model_meta COMMAND test_model_meta)
add_executable(test_detector_load tests/test_detector_load.cpp)
target_link_libraries(test_detector_load PRIVATE ppe_core)
add_test(NAME detector_load COMMAND test_detector_load)
add_executable(test_detector_decode tests/test_detector_decode.cpp)
target_link_libraries(test_detector_decode PRIVATE ppe_core)
add_test(NAME detector_decode COMMAND test_detector_decode)
add_executable(test_detector_filter tests/test_detector_filter.cpp)
target_link_libraries(test_detector_filter PRIVATE ppe_core)
add_test(NAME detector_filter COMMAND test_detector_filter)
add_executable(test_detector_nms tests/test_detector_nms.cpp)
target_link_libraries(test_detector_nms PRIVATE ppe_core)
add_test(NAME detector_nms COMMAND test_detector_nms)
add_executable(test_debug_draw tests/test_debug_draw.cpp)
target_link_libraries(test_debug_draw PRIVATE ppe_core)
add_test(NAME debug_draw COMMAND test_debug_draw)

# MSVC가 소스 파일을 UTF-8로 읽도록 설정 (한글 주석·문자열 깨짐 방지)
if(MSVC)
  foreach(t ppe_core safety_app test_video_source test_letterbox test_monitor_window test_pipeline test_model_meta test_detector_load test_detector_decode test_detector_filter test_detector_nms test_debug_draw)
    target_compile_options(${t} PRIVATE /utf-8)
  endforeach()
endif()

# DET-05 ONNX Runtime 엔진 (선택)
# SDK가 있는 PC만 CMakeUserPresets.json에 ONNXRUNTIME_ROOT를 지정함. 없으면 이 엔진만 빼고 빌드함
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime 1.20.0 NuGet 패키지를 푼 폴더")
if(ONNXRUNTIME_ROOT)
  set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/buildTransitive/native/include")
  set(ORT_NATIVE_DIR "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native")
  if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h" OR NOT EXISTS "${ORT_NATIVE_DIR}/onnxruntime.lib")
    message(FATAL_ERROR "ONNXRUNTIME_ROOT에서 ONNX Runtime SDK를 찾을 수 없음: ${ONNXRUNTIME_ROOT}")
  endif()

  add_library(ppe_ort STATIC src/detect/OrtDetector.cpp)
  target_include_directories(ppe_ort PUBLIC "${ORT_INCLUDE_DIR}")
  target_link_libraries(ppe_ort PUBLIC ppe_core "${ORT_NATIVE_DIR}/onnxruntime.lib")

  add_executable(test_ort_detector tests/test_ort_detector.cpp)
  target_link_libraries(test_ort_detector PRIVATE ppe_ort)
  add_test(NAME ort_detector COMMAND test_ort_detector)

  # 파이프라인 실제 실행 테스트에서도 ONNX Runtime 엔진을 고를 수 있게 함
  target_link_libraries(test_pipeline PRIVATE ppe_ort)
  target_compile_definitions(test_pipeline PRIVATE PPE_WITH_ORT)
  add_dependencies(test_pipeline test_ort_detector)

  if(MSVC)
    target_compile_options(ppe_ort PRIVATE /utf-8)
    target_compile_options(test_ort_detector PRIVATE /utf-8)
  endif()

  # Windows 기본 폴더의 다른 버전 onnxruntime.dll이 쓰이지 않도록 exe 옆에 복사함
  add_custom_command(TARGET test_ort_detector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_NATIVE_DIR}/onnxruntime.dll"
            "${ORT_NATIVE_DIR}/onnxruntime_providers_shared.dll"
            "${ORT_NATIVE_DIR}/onnxruntime_providers_cuda.dll"
            $<TARGET_FILE_DIR:test_ort_detector>)
endif()

# Windows에서 빌드 후 OpenCV DLL을 exe와 같은 폴더로 복사 (PATH 환경 변수 설정 불필요)
if(WIN32)
  # 링크된 OpenCV DLL (예: opencv_world4xx.dll, Debug 구성은 opencv_world4xxd.dll)
  add_custom_command(TARGET safety_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:safety_app> $<TARGET_FILE_DIR:safety_app>
    COMMAND_EXPAND_LISTS)

  # 영상 파일 읽기용 FFmpeg 플러그인 DLL (링크 대상이 아니라 실행 중에 불러오므로 따로 복사)
  # OpenCV_DIR을 build 폴더로 지정한 경우와 x64/vcXX/lib 폴더로 지정한 경우를 모두 찾음
  file(GLOB OPENCV_FFMPEG_DLLS
       "${OpenCV_DIR}/x64/*/bin/opencv_videoio_ffmpeg*.dll"
       "${OpenCV_DIR}/../bin/opencv_videoio_ffmpeg*.dll")
  if(OPENCV_FFMPEG_DLLS)
    add_custom_command(TARGET safety_app POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${OPENCV_FFMPEG_DLLS} $<TARGET_FILE_DIR:safety_app>)
  endif()
endif()