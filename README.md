## 개요
- 안드로이드 기기에서 cctv를 스트리밍하고 USE 디버깅을 통해 안드로이드 기기 화면을 PC로 미러링 시킨다.
- PC화면을 20 fps로 10장 추출하여 각각 raw RGB 프레임 형태로 리스트에 넣은 후 sma3 모델로 입력된다. 0.5 간격으로 리스트가 입력되며 준실시간으로 데이터가 입력된다.

## 파일 설명
- live_demo.py는 sam3 모델에 테스트해본 파일이다.
- frame_maker.py는 원하는 모델에 frame 리스트를 입력할 수 있는 파이프라인이다.

## 필요 설치물
- USB 디버깅을 위한 ADB (Android platform-tools), 실시간 미러링을 위한 scrcpy, 실시간 프레임 추출을 위한 ffmpeg 설치해야한다.

## Chocolatey 설치
- 먼저 편리한 설치를 위해 powershell "관리자 권한으로 실행" 후 Chocolatey을 설치한다
- Chocolatey 설치

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

## scrcpy 설치

```powershell
choco install scrcpy -y
```

## adb 설치

```powershell
choco install adb -y
```

## ffmpeg 설치

```powershell
choco install ffmpeg -y
```

## 실행
- *안드로이드 기기기 화면을 실시간으로 PC에 스트리밍
  - 안드로이드 기기 개발자 모드 설정 및 USE 디버깅 허용 후 PC와 연결결
  - powershell에서 아래 명령어 실행

```powershell
scrcpy --max-size 1280 --video-bit-rate 4M --max-fps 30 --no-audio
```

- *live_demo.py 실행
  - powershell에서 아래 명령어 실행

```powershell
python live_demo.py --fps 20 --width 640 --height 360 --batch 10 --device cuda
```

- *frame_maker.py 실행

```powershell
python frame_maker.py --fps 20 --width 640 --height 360 --batch 10 --print-only
```

- **fp3와 batch변경 가능

## 속도 공식(직관적으로)
- “추론 갱신 주기(초)” = batch / fps
- 예: fps=20, batch=10 -> 0.5초마다 추론

## ffmpeg 캡처 주의사항(중요)
- 지금 방식은 desktop 캡처라서 scrcpy 창이 다른 창에 가리면 그대로 캡처됨 -> “항상 최상단” 권장
