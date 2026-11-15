@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo       Kompilacja Path Tracer 3D (CPU, CUDA, Win32 GUI)
echo ========================================================

set LLVM_BIN=C:\Users\Michał\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin
set CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64

if exist "%LLVM_BIN%\clang++.exe" (
    set CXX="%LLVM_BIN%\clang++.exe"
) else (
    set CXX=clang++
)

echo [1/3] Kompilacja CPU Path Tracer (pathtracer.exe)...
%CXX% -std=c++17 -O3 -static main.cpp -Iinclude -o pathtracer.exe
if %ERRORLEVEL% neq 0 (
    echo [BLAD] Nie udalo sie skompilowac pathtracer.exe!
    pause
    exit /b %ERRORLEVEL%
)
echo [OK] pathtracer.exe gotowy.

echo.
echo [2/3] Kompilacja GPU CUDA Path Tracer (pathtracer_cuda.exe)...
%CXX% -std=c++17 -O3 -static pathtracer_cuda.cpp cuda_pt.cpp -Iinclude -I"%CUDA_INC%" -L. -L"%CUDA_LIB%" -lcuda -lnvrtc -o pathtracer_cuda.exe
if %ERRORLEVEL% neq 0 (
    echo [BLAD] Nie udalo sie skompilowac pathtracer_cuda.exe!
    pause
    exit /b %ERRORLEVEL%
)
echo [OK] pathtracer_cuda.exe gotowy.

echo.
echo [3/3] Kompilacja Glownej Aplikacji GUI (pathtracer_gui.exe)...
%CXX% -std=c++17 -O3 -static gui_main.cpp cuda_pt.cpp -Iinclude -I"%CUDA_INC%" -L. -L"%CUDA_LIB%" -lgdi32 -luser32 -lcomctl32 -lcomdlg32 -lOpenImageDenoise -lcuda -lnvrtc -mwindows -o pathtracer_gui.exe
if %ERRORLEVEL% neq 0 (
    echo [BLAD] Nie udalo sie skompilowac pathtracer_gui.exe!
    pause
    exit /b %ERRORLEVEL%
)
echo [OK] pathtracer_gui.exe gotowy.

echo.
echo ========================================================
echo       Wszystkie programy zostaly pomyslnie zbudowane!
echo ========================================================
pause
