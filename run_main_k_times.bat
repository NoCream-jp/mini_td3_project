@echo off
setlocal

rem ===== 実行するファイルを固定 =====
set "SCRIPT=C:./main.py"

rem ===== 実行回数の入力 =====
set /p "COUNT=How many times to run? (e.g. 5): "

rem ===== 入力チェック =====
echo(%COUNT%|findstr /R "^[1-9][0-9]*$" >nul
if errorlevel 1 (
    echo [ERROR] Please enter a positive integer.
    pause
    exit /b 1
)

rem ===== ファイルの存在確認 =====
if not exist "%SCRIPT%" (
    echo [ERROR] File not found: "%SCRIPT%"
    pause
    exit /b 1
)

rem ===== main.pyのフォルダに移動 =====
for %%F in ("%SCRIPT%") do pushd "%%~dpF"
if errorlevel 1 exit /b 1

rem ===== COUNT回実行 =====
for /L %%I in (1,1,%COUNT%) do (
    echo.
    echo ===== Run %%I / %COUNT% =====

    python "%SCRIPT%"

    if errorlevel 1 goto failed
)

echo.
echo All %COUNT% runs completed.
popd
pause
exit /b 0

:failed
echo [ERROR] Execution failed.
popd
pause
exit /b 1
