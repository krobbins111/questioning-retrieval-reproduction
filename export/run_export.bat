@echo off
rem Rebuild the released answers from the original runs (authors only: the runs themselves are not released).
rem    export\run_export.bat                       every source, the side tables, the larger question set, the steered
rem                                                search (a few minutes over SigLIP2's 49 shards), the dataset card, then the scan
rem    export\run_export.bat --only docci_s1       one source (the side tables and the card are rewritten too)
rem The runs: ..\jevrank next to this repository. INQUIRE-Fullrank (inquire_full), the DOCCI sentence parts
rem (docci_parts) and the output (questioning-retrieval-data) sit under %QR_ROOT%, default C:\data.
rem Needs numpy and pyarrow in the Python it finds.
setlocal
set HERE=%~dp0..
if "%QR_ROOT%"=="" set QR_ROOT=C:\data
set PY=%HERE%\..\venv\Scripts\python.exe
if not exist "%PY%" set PY=python
set OUT=%QR_ROOT%\questioning-retrieval-data
"%PY%" "%HERE%\export\export_answers.py" --jevrank "%HERE%\..\jevrank" --inquire-full "%QR_ROOT%\inquire_full" --out "%OUT%" %* || exit /b 1
"%PY%" "%HERE%\export\export_extras.py" --jevrank "%HERE%\..\jevrank" --inquire-full "%QR_ROOT%\inquire_full" --docci-parts "%QR_ROOT%\docci_parts" --out "%OUT%" || exit /b 1
"%PY%" "%HERE%\export\export_bank.py" --jevrank "%HERE%\..\jevrank" --inquire-full "%QR_ROOT%\inquire_full" --out "%OUT%" || exit /b 1
"%PY%" "%HERE%\export\export_steer.py" --jevrank "%HERE%\..\jevrank" --inquire-full "%QR_ROOT%\inquire_full" --out "%OUT%" --state "%TEMP%\qr_steer" || exit /b 1
"%PY%" "%HERE%\export\dataset_card.py" --out "%OUT%" || exit /b 1
copy /y "%OUT%\manifest.json" "%HERE%\configs\data_manifest.json" >nul
"%PY%" "%HERE%\scripts\anon_scan.py" --data "%OUT%" || exit /b 1
"%PY%" "%HERE%\scripts\fetch_data.py" --verify || exit /b 1
endlocal
