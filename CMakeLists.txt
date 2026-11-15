cmake_minimum_required(VERSION 3.14)
project(PathTracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
)
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64"
)

# 1. Wersja konsolowa CPU
add_executable(pathtracer main.cpp)

# 2. Wersja konsolowa GPU (NVIDIA CUDA Path Tracer)
add_executable(pathtracer_cuda pathtracer_cuda.cpp cuda_pt.cpp)
target_link_libraries(pathtracer_cuda PRIVATE cuda nvrtc)

# 3. Główna aplikacja okienkowa (Win32 GUI z obsługą CPU + GPU CUDA)
add_executable(pathtracer_gui WIN32 gui_main.cpp cuda_pt.cpp)
target_link_libraries(pathtracer_gui PRIVATE gdi32 user32 comctl32 comdlg32 OpenImageDenoise cuda nvrtc)

if(MSVC)
    target_compile_options(pathtracer PRIVATE /O2 /W4)
    target_compile_options(pathtracer_cuda PRIVATE /O2 /W4)
    target_compile_options(pathtracer_gui PRIVATE /O2 /W4)
else()
    target_compile_options(pathtracer PRIVATE -O3 -Wall -static)
    target_compile_options(pathtracer_cuda PRIVATE -O3 -Wall -static)
    target_compile_options(pathtracer_gui PRIVATE -O3 -Wall -static)
endif()
