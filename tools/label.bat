@echo off
rem Possession labelling in mpv (keys: h / a / d at each change, x to close; see label_possession.lua).
rem   tools\label.bat [start] [output csv]     defaults: 25:00 and validation/intervals.csv
rem   tools\label.bat 7:27.4 validation/calibration.csv     calibration on the public window
cd /d "%~dp0.."
set "START=%~1"
if "%START%"=="" set "START=25:00"
set "OUT=%~2"
if "%OUT%"=="" set "OUT=validation/intervals.csv"
set "MPV=C:\Program Files\MPV Player\mpv.exe"
where mpv >nul 2>nul && set "MPV=mpv"
"%MPV%" --script=tools/label_possession.lua --script-opts=label-out=%OUT% --osd-fractions --start=%START% "data\tactical\BAR-ATH_tactical.mp4"
