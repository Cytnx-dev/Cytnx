@echo off
rem Pixi runs this script while capturing the activation environment, and
rem tasks may invoke it again with a command. Reuse a matching prefix only
rem when its compiler and Python are actually still reachable: Pixi can cache
rem ordinary variables independently of PATH after an environment update.
if not "%~1"=="" if defined CONDA_PREFIX if /i "%CYTNX_WINDOWS_ACTIVATED_PREFIX%"=="%CONDA_PREFIX%" (
  where.exe cl >nul 2>&1
  if not errorlevel 1 where.exe python >nul 2>&1
  if not errorlevel 1 goto run_command
)

rem Prefix-derived paths must be set by an activation script. Pixi's
rem activation.env values are literal and do not expand %%CONDA_PREFIX%%.
set "CYTNX_VS_INSTALL="
for /f "usebackq tokens=*" %%i in (`"%CONDA_PREFIX%\Library\bin\vswhere.exe" -nologo -latest -products * -version [17.0^,18.0^) -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "CYTNX_VS_INSTALL=%%i"
if not defined CYTNX_VS_INSTALL (
  echo Cytnx requires Visual Studio 2022 with the Desktop development with C++ workload. 1>&2
  exit /b 1
)
call "%CYTNX_VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
set "CYTNX_VS_INSTALL="

rem vcvars64 prepends MSVC; keep it while restoring Pixi's managed tools and
rem dependency DLLs ahead of the ambient user PATH.
set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Library\mingw-w64\bin;%CONDA_PREFIX%\Library\usr\bin;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\bin;%PATH%"
set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Library"
set "MKLROOT=%CONDA_PREFIX%\Library"

set "CYTNX_CUDA_PREFIX=%CONDA_PREFIX%\Lib\site-packages\nvidia\cu13"
if exist "%CYTNX_CUDA_PREFIX%\bin\nvcc.exe" (
  set "CUDA_PATH=%CYTNX_CUDA_PREFIX%"
  set "CUDA_HOME=%CYTNX_CUDA_PREFIX%"
  set "CUDAToolkit_ROOT=%CYTNX_CUDA_PREFIX%"
  set "CUDACXX=%CYTNX_CUDA_PREFIX%\bin\nvcc.exe"
  set "CUTENSOR_ROOT=%CONDA_PREFIX%\Lib\site-packages\cutensor"
  set "PATH=%CYTNX_CUDA_PREFIX%\bin;%CYTNX_CUDA_PREFIX%\bin\x64;%CYTNX_CUDA_PREFIX%\bin\x86_64;%CONDA_PREFIX%\Lib\site-packages\cutensor\bin;%CONDA_PREFIX%\Lib\site-packages\cutensor\lib;%PATH%"
)
set "CYTNX_CUDA_PREFIX="
set "CYTNX_WINDOWS_ACTIVATED_PREFIX=%CONDA_PREFIX%"

:run_command
rem When a command is supplied, execute it in this activated batch process so
rem the Visual Studio and dependency search paths stay in the same process.
if /i "%~1"=="--doctor" (
  where.exe cl || exit /b 1
  python --version || exit /b 1
  cmake --version || exit /b 1
  ninja --version || exit /b 1
  exit /b 0
)
if /i "%~1"=="--cuda-doctor" (
  where.exe cl || exit /b 1
  if not defined CUDACXX exit /b 1
  "%CUDACXX%" --version || exit /b 1
  python --version || exit /b 1
  cmake --version || exit /b 1
  exit /b 0
)
if "%~1"=="" exit /b 0
call %*
exit /b %errorlevel%
