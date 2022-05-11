@echo off
git add .
set /p msg=input the commit msg:
git commit -m %msg%
git push origin master
pause