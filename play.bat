@echo off
REM Launch the Cute Chess GUI to play against Checksmith.
REM The first time, add the engine: Tools > Settings > Engines > Add,
REM browse to target\release\checksmith.exe, protocol UCI. See PLAYING.md.
start "" "%~dp0tools\cutechess-1.4.0-win64\cutechess.exe"
