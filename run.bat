@echo off
echo ========================================
echo YouTube Ad Recommendation System
echo ========================================
echo.

:menu
echo Select an option:
echo 1. Install Dependencies
echo 2. Run EDA (Exploratory Data Analysis)
echo 3. Train Model
echo 4. Generate Report (Plots + CSVs)
echo 5. Test Predictions
echo 6. Start Web Application
echo 7. Exit
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto eda
if "%choice%"=="3" goto train
if "%choice%"=="4" goto test
if "%choice%"=="5" goto predict
if "%choice%"=="6" goto webapp
if "%choice%"=="7" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:install
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Dependencies installed successfully!
pause
goto menu

:eda
echo.
echo Running Exploratory Data Analysis...
python EDA.py
echo.
pause
goto menu

:train
echo.
echo Training the model...
python train.py
echo.
pause
goto menu

:test
echo.
echo Generating comprehensive report...
echo This will create plots and CSVs in output folder
python test.py
echo.
pause
goto menu

:predict
echo.
echo Running predictions...
python predict.py
echo.
pause
goto menu

:webapp
echo.
echo Starting web application...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
python app.py
pause
goto menu

:end
echo.
echo Goodbye!
exit
