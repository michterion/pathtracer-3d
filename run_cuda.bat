@echo off
cd /d "%~dp0"
echo Uruchamianie NVIDIA CUDA Path Tracer (GPU)...
pathtracer_cuda.exe --width 1280 --height 720 --spp 256 --depth 32 --out output_cuda.png
pause
