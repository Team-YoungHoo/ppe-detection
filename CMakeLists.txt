cmake_minimum_required(VERSION 3.21)
project(safety_app CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
add_executable(safety_app src/main.cpp)
target_link_libraries(safety_app ${OpenCV_LIBS})

# MSVC가 소스 파일을 UTF-8로 읽도록 설정 (한글 주석·문자열 깨짐 방지)
if(MSVC)
  target_compile_options(safety_app PRIVATE /utf-8)
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