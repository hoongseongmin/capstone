# C:\Users\tjdal\Desktop\capstone\start_app.bat
@echo off
echo 소비지출데이터 기반 거래내역분석 시스템을 시작합니다...
cd /d C:\Users\tjdal\Desktop\capstone
call .\venv\Scripts\activate.bat
uvicorn app.main:app --reload