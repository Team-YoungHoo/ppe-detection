cmake_minimum_required(VERSION 3.21)
project(safety_app CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

# 모듈 코드 (main.cpp를 뺀 나머지)
# safety_app과 테스트가 같이 씀
add_library(ppe_core STATIC
  src/io/VideoSource.cpp
  src/preprocess/Letterbox.cpp
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

# MSVC가 소스 파일을 UTF-8로 읽도록 설정 (한글 주석·문자열 깨짐 방지)
if(MSVC)
  foreach(t ppe_core safety_app test_video_source test_letterbox)
    target_compile_options(${t} PRIVATE /utf-8)
  endforeach()
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