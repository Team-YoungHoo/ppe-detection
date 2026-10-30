# 건설현장 작업자 안전장비 미착용 감지 시스템

CCTV 영상에서 작업자의 안전모·안전조끼 미착용을 감지하고 경고하는 C++ 프로그램입니다.
(숙명여자대학교 2026-2 컴퓨터비전 캡스톤디자인, 팀 영후)

---

## 1. 필요한 것

| 항목 | 내용 |
| --- | --- |
| 운영체제 | Windows (Mac·Linux는 [6. 참고](#6-참고-maclinux에서-빌드) 참고) |
| Visual Studio | **"C++를 사용한 데스크톱 개발"** 워크로드 설치 ("Windows용 C++ CMake 도구" 포함) |
| CMake | 3.21 이상 (Visual Studio에 포함된 CMake 사용) |
| OpenCV | OpenCV 4.x Windows 배포판. **설치 위치는 자유** |

---

## 2. 처음 한 번만: OpenCV 위치 알려주기

OpenCV 설치 위치는 사람마다 다르므로, 아래 **A, B 중 하나**를 골라 한 번만 설정합니다.
지정할 경로는 `OpenCVConfig.cmake` 파일이 있는 폴더입니다. (보통 `<OpenCV 설치 위치>/opencv/build`)

### 방법 A (권장): `CMakeUserPresets.json` 만들기

이 프로젝트에만 적용되고, 시스템 설정을 바꾸지 않습니다.

1. 저장소 루트(`CMakePresets.json`이 있는 폴더)에 `CMakeUserPresets.json` 파일을 만듭니다.
2. 아래 내용을 붙여 넣고, `OpenCV_DIR` 두 곳을 **본인 경로**로 바꿉니다. 경로 구분자는 `/`를 씁니다.

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "my-debug",
      "displayName": "My Debug",
      "inherits": "x64-debug",
      "cacheVariables": { "OpenCV_DIR": "C:/본인경로/opencv/build" }
    },
    {
      "name": "my-release",
      "displayName": "My Release",
      "inherits": "x64-release",
      "cacheVariables": { "OpenCV_DIR": "C:/본인경로/opencv/build" }
    }
  ]
}
```

> 이 파일은 개인 설정이라 `.gitignore`에 들어 있습니다. **커밋하지 마세요.**

### 방법 B: 사용자 환경 변수 `OpenCV_DIR` 설정

1. Windows 검색 → "시스템 환경 변수 편집" → **환경 변수**
2. 위쪽 **사용자 변수**에서 **새로 만들기**
   - 변수 이름: `OpenCV_DIR`
   - 변수 값: `C:\본인경로\opencv\build`
3. Visual Studio를 **다시 시작**합니다.

---

## 3. 빌드

1. 저장소를 clone 합니다.
   ```bash
   git clone https://github.com/Team-YoungHoo/<저장소 이름>.git
   ```
2. Visual Studio → **파일 → 열기 → 폴더** → clone한 폴더를 엽니다.
3. 상단 구성 드롭다운에서 구성을 고릅니다.
   - 방법 A: **My Release** (디버깅할 때는 My Debug)
   - 방법 B: **x64 Release** (디버깅할 때는 x64 Debug)
4. **빌드 → 모두 빌드**

빌드가 끝나면 `out/build/<구성 이름>/` 폴더에 `safety_app.exe`와 OpenCV DLL이 함께 생깁니다.
DLL은 빌드할 때 자동으로 복사되므로 `PATH` 설정이 필요 없습니다.

> 처리 속도(FPS) 측정은 반드시 **Release** 구성으로 합니다. Debug는 훨씬 느립니다.

---

## 4. 실행

실행할 때 영상 파일 경로를 넣어야 합니다.

**Visual Studio에서 실행**

1. 상단 시작 항목에서 `safety_app.exe` 선택
2. 시작 항목 옆 메뉴 → **디버그 및 시작 설정** → 열리는 `launch.vs.json`의 해당 항목에 추가
   ```json
   "args": [ "C:/videos/test.mp4" ]
   ```
3. 실행 (F5 또는 Ctrl+F5)

**터미널에서 실행**

```bash
out\build\my-release\safety_app.exe C:\videos\test.mp4
```

- 종료: `ESC` 또는 `q` 키, 창 닫기, 영상 끝
- 종료 코드: `0` 정상, `2` 입력 영상 오류

> 영상 경로에는 한글·공백이 없는 것이 안전합니다.

---

## 5. 문제 해결

| 증상 | 확인할 것 |
| --- | --- |
| 구성 단계에서 `Could not find a package configuration file provided by "OpenCV"` | 2단계 설정 확인. 경로가 `OpenCVConfig.cmake`가 있는 폴더인지, 방법 A라면 My 구성을 골랐는지, 방법 B라면 Visual Studio를 다시 시작했는지 |
| 구성 단계에서 CMake 버전 오류 | Visual Studio를 업데이트해서 CMake 3.21 이상 사용 |
| 실행 시 `opencv_world*.dll`이 없다는 오류 | 실행하는 exe와 같은 폴더에 DLL이 있는지 확인. 없으면 다시 빌드 |
| `[ERROR] Cannot open video` | 영상 경로 확인(한글·공백 제외). exe 폴더에 `opencv_videoio_ffmpeg*.dll`이 있는지 확인 |

---

## 6. 참고: Mac·Linux에서 빌드

OpenCV를 패키지 관리자로 설치한 경우 별도 경로 설정 없이 빌드됩니다.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/safety_app <영상 경로>
```