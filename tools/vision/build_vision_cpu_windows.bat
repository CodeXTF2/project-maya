@echo off
rem strata-vision (llama.cpp mtmd + GLM-5.3-Flash's vision tower) for the CPU, with MSVC - the image encoder Maya's
rem server runs beside the HIP engine ("vision" entry in the config, "gpu": false).  Run from anywhere; the build
rem goes to build-vision\bin\strata-vision.exe at the repository's root.
set "ROOT=%~dp0..\..\"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
"%ROOT%.venv\Scripts\python.exe" "%ROOT%tools\vision\glm5next_patch.py" "%ROOT%third_party\llama.cpp" || exit /b 2
"%ROOT%.venv\Scripts\cmake.exe" -G Ninja -DCMAKE_MAKE_PROGRAM="%ROOT%.venv\Scripts\ninja.exe" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -S "%ROOT%tools\vision" -B "%ROOT%build-vision" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="%ROOT%third_party\llama.cpp" -DSTRATA_VISION_CUDA=OFF || exit /b 3
"%ROOT%.venv\Scripts\cmake.exe" --build "%ROOT%build-vision" --target strata-vision -j 8 || exit /b 4
