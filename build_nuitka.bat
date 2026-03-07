@echo off
chcp 65001
echo ====================================
echo Nuitka 打包專案 (DirectML ONNX 版)
echo ====================================
echo.
echo 🚀 針對 DirectML 優化的 ONNX 打包方案
echo    - 使用 ONNX Runtime DirectML
echo    - 不包含 PyTorch (已改用 ONNX)
echo    - 無 PyQt6 插件，避免 tkinter 衝突
echo.

echo [1/6] 激活虛擬環境...
call .\venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo 錯誤: 無法激活虛擬環境
    pause
    exit /b 1
)

echo.
echo [2/6] 檢查 Python 路徑和依賴...
for /f "delims=" %%i in ('python -c "import sys; print(sys.prefix)"') do set PYTHON_PATH=%%i
echo Python 路徑: %PYTHON_PATH%

set ONNX_PATH=%PYTHON_PATH%\Lib\site-packages\onnxruntime

if not exist "%ONNX_PATH%" (
    echo ❌ 錯誤: 找不到 onnxruntime 路徑
    pause
    exit /b 1
)

echo ✅ ONNX Runtime 路徑: %ONNX_PATH%
echo.

echo [3/6] 檢查 Nuitka...
pip show nuitka >nul 2>&1
if %errorlevel% neq 0 (
    echo Nuitka 未安裝，正在安裝...
   pip install nuitka
) else (
    echo Nuitka 已安裝
)

echo.
echo [4/6] 清理舊的建構檔案...
if exist build rmdir /s /q build
if exist main.build rmdir /s /q main.build
if exist main.dist rmdir /s /q main.dist
if exist main.onefile-build rmdir /s /q main.onefile-build

echo.
echo [5/6] 開始 Nuitka 打包 (無 PyQt6 插件模式)...
echo 自動追踪導入（包含 tkinter），預計時間: 10-15 分鐘
echo.

python -m nuitka ^
    --standalone ^
    --mingw64 ^
    --enable-plugin=pyqt6 ^
    --enable-plugin=tk-inter ^
    --include-module=pynput.keyboard._win32 ^
    --include-module=pynput.mouse._win32 ^
    --include-module=pynput._util.win32 ^
    --nofollow-import-to=torch ^
    --nofollow-import-to=ultralytics ^
    --nofollow-import-to=IPython ^
    --nofollow-import-to=pandas ^
    --include-data-dir=Module=Module ^
    --include-data-dir="%ONNX_PATH%=onnxruntime" ^
    --include-data-dir=ui=ui ^
    --include-data-file=app.ico=app.ico ^
    --include-data-file=dll64.dll=dll64.dll ^
    --include-data-file=hiddll_x64.dll=hiddll_x64.dll ^
    --include-data-file=qt.conf=qt.conf ^
    --include-package-data=onnxruntime ^
    --follow-imports ^
    --lto=yes ^
    --windows-icon-from-ico=app.ico ^
    --windows-console-mode=force ^
    --output-dir=build ^
    --show-progress ^
    --show-memory ^
    --assume-yes-for-downloads ^
    --jobs=%NUMBER_OF_PROCESSORS% ^
    main.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ 打包失敗！
    pause
    exit /b 1
)

echo.
echo [6/6] 後續處理：複製資料夾和創建啟動腳本...
echo.

:: 複製 Data 和 Model 資料夾
if exist Data (
    echo 複製 Data 資料夾...
    xcopy /E /I /Y Data build\main.dist\Data >nul
    echo ✅ Data 已複製
)

if exist Model (
    echo 複製 Model 資料夾...
    xcopy /E /I /Y Model build\main.dist\Model >nul
    echo ✅ Model 已複製
)

:: DirectML 特殊處理：複製關鍵 DLL
echo.
echo 🔧 DirectML 特殊處理...
if exist "build\main.dist\onnxruntime\capi" (
    echo    - 複製 ONNX Runtime DLL...
    xcopy /Y "build\main.dist\onnxruntime\capi\*.dll" "build\main.dist\" >nul 2>&1
)

:: 查找並複製 DirectML.dll
for /r "%PYTHON_PATH%\Lib\site-packages" %%f in (DirectML.dll) do (
    if exist "%%f" (
        echo    - 複製 DirectML.dll...
        copy /Y "%%f" "build\main.dist\" >nul 2>&1
        echo    ✅ DirectML.dll 已複製
    )
)

:: 創建啟動腳本
echo    - 創建啟動腳本...
(
echo @echo off
echo cd /d "%%~dp0"
echo set PATH=%%CD%%\onnxruntime\capi;%%PATH%%
echo start main.exe
) > "build\main.dist\run.bat"
echo ✅ run.bat 已創建

:: 創建 README
(
echo ====================================
echo AI AIMBOT - DirectML ONNX 版本
echo ====================================
echo.
echo 🎮 此版本使用 ONNX Runtime DirectML 加速
echo    支援 AMD/Intel/NVIDIA GPU
echo    已移除 PyTorch 依賴，體積更小
echo.
echo 📁 檔案結構:
echo    - main.exe           : 主程式
echo    - run.bat            : 推薦啟動方式
echo    - Module\            : 程式模組
echo    - Data\              : 數據資料夾
echo    - Model\             : AI 模型 ^(ONNX 格式^)
echo    - onnxruntime\       : ONNX Runtime ^(DirectML^)
echo    - DirectML.dll       : DirectML 加速庫
echo    - tkinter\           : 鍵盤捕獲功能
echo.
echo 🚀 啟動方式:
echo    方式 1 ^(推薦^): 雙擊 run.bat
echo    方式 2: 直接運行 main.exe
echo.
echo ⚠️  系統需求:
echo    1. Windows 10 1709 或更高版本
echo    2. DirectX 12 相容的 GPU
echo    3. Visual C++ Redistributable 2015-2022
echo       下載: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
echo 💡 如果程式無法啟動:
echo    1. 確認 DirectX 12 已安裝
echo    2. 確認 GPU 驅動是最新版本
echo    3. 使用 run.bat 啟動
echo    4. 查看是否有錯誤訊息
echo.
echo 🔍 DirectML 支援:
echo    - NVIDIA GPU: GTX 900 系列或更新
echo    - AMD GPU: GCN 架構或更新
echo    - Intel GPU: Gen 9 或更新
echo.
echo 📝 版本說明:
echo    - 使用 ONNX Runtime 取代 PyTorch
echo    - 打包體積更小，啟動更快
echo    - 需要 ONNX 格式的模型文件
echo    - 無 PyQt6 插件，避免 tkinter 衝突
echo.
) > "build\main.dist\README_DirectML.txt"
echo ✅ README_DirectML.txt 已創建

echo.
echo ====================================
echo ✅ DirectML ONNX 版本打包完成！
echo ====================================
echo.
echo 📁 可執行檔案位於: build\main.dist\
echo 📦 總大小: 
dir /s build\main.dist 2>nul | find "File(s)"
echo.
echo 🎯 已完成的工作:
echo    ✅ 1. 包含 ONNX Runtime DirectML
echo    ✅ 2. 包含 DirectML.dll
echo    ✅ 3. 自動追踪所有導入模組 (包括 tkinter)
echo    ✅ 4. 包含 Module/Data/Model 資料夾
echo    ✅ 5. 創建啟動腳本 run.bat
echo    ✅ 6. 複製關鍵 DLL 到主目錄
echo    ✅ 7. 生成說明文檔
echo    ✅ 8. 已排除 PyTorch/Ultralytics
echo    ✅ 9. 無 PyQt6 插件，避免 tkinter 衝突
echo.
echo 📝 配置說明:
echo    - 移除了 --enable-plugin=pyqt6
echo    - --follow-imports 會自動追踪 PyQt6 和 tkinter
echo    - 這樣可以避免插件對 tkinter 的排除
echo.
echo 🧪 測試步驟:
echo    1. cd build\main.dist
echo    2. 雙擊 run.bat 或執行 main.exe
echo    3. 確認 DirectML GPU 加速正常運作
echo    4. 測試鍵盤捕獲功能是否正常
echo.
echo 📤 發布到其他電腦:
echo    1. 複製整個 main.dist 資料夾
echo    2. 確保目標電腦有 DirectX 12
echo    3. 安裝 VC++ Redistributable ^(如需要^)
echo    4. 使用 run.bat 啟動
echo    5. 確保使用 ONNX 格式的模型文件
echo.
pause
