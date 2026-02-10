@echo off
echo ========================================
echo   SCALP RADAR - Dev Environment
echo ========================================

:: Se placer dans le repertoire du script
cd /d "%~dp0"

:: Verifier que .venv existe
if not exist ".venv" (
    echo [!] .venv absent, lancement de uv sync...
    uv sync
)

:: Verifier que node_modules existe
if not exist "frontend\node_modules" (
    echo [!] node_modules absent, lancement de npm install...
    pushd frontend
    npm install
    popd
)

:: Backend (FastAPI + DataEngine via lifespan)
echo [*] Lancement du backend (port 8000)...
start "SCALP-BACKEND" cmd /k "cd /d %~dp0 && uv run uvicorn backend.api.server:app --reload --port 8000"

:: Frontend (Vite)
echo [*] Lancement du frontend (port 5173)...
start "SCALP-FRONTEND" cmd /k "cd /d %~dp0frontend && npx vite"

echo.
echo [OK] Backend : http://localhost:8000
echo [OK] Frontend: http://localhost:5173
echo [OK] Health  : http://localhost:8000/health
echo.
echo Pour desactiver le WebSocket en dev:
echo   set ENABLE_WEBSOCKET=false avant de lancer
