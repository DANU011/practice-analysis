# practice-analysis

> [`R`](https://www.r-project.org/)에서 진행된 [회귀 분석](https://en.wikipedia.org/wiki/Regression_analysis), [분류](classification), [시계열](https://en.wikipedia.org/wiki/Time_series) 분석 방법을 파이썬으로 변경하여 진행하는 과정을 연습한 자료를 저장해 둔 자료입니다. R에서 진행된 분석 방법은 `with_R` 폴더의 `PDF` 파일을 확인하세요.

## 설정
* Python <= `3.10.10`
    * `numpy`, `pandas`, `matplotlib`는 데이터 분석을 사용
    * 고급 시각화를 위해서 `plotly`도 병행해서 사용
    * 회귀 분석을 위해서 `xgboost`, `Optuna`를 활용
    * `ipynb`를 위해서 `JupyterLab`도 함께 설치    
    * `requirements.txt` 파일을 참고해서 설치

```shell
$ python --version
Python 3.10.11
$ python -m venv venv
# for Windows
$ .\venv\Scripts\activate
# for Linux/macOS
$ source ./venv/bin/activate
$ (venv) pip install -r requirements.txt
```

## 사용된 패키지 목록
* numpy
* pandas
* seaborn
* plotly
* jupyterlab
* statsmodels
* scikit-learn
* xgboost
* optuna

## ToDo

- [ ] 회귀 문제([캘리포니아 집 값 데이터](http://lib.stat.cmu.edu/datasets/))
- [ ] 분류 문제([타이타닉 생존자 예측](https://www.kaggle.com/c/titanic))
- [ ] 시계열 문제(문제 탐색 중)
