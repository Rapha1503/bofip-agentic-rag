@echo off
title BOFIP Agentic RAG
echo.
echo  BOFIP Agentic RAG
echo  -----------------
echo.
cd /d "%~dp0"
call .\venv\Scripts\activate.bat 2>nul
streamlit run app.py --server.port 8501
pause
