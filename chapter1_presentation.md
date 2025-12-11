# Chapter 1: 계량경제학과 Python: 시작하기

---

## 1.1 계량경제학이란?

### 계량경제학의 정의
- 경제 이론을 실증적으로 검증하는 학문
- 통계학적 방법론을 경제 데이터에 적용
- 경제 현상의 정량적 분석과 예측

### 왜 Python인가?
- 무료 오픈소스 생태계
- 풍부한 데이터 분석 라이브러리
- 학계와 산업계에서 널리 사용
- 배우기 쉽고 강력한 기능

---

## 1.2 Python 도구 셋업

### 필수 설치 항목
1. **Python 3.8 이상**
2. **Anaconda 또는 Miniconda**
3. **Jupyter Notebook/Lab**

### 설치 방법
```bash
# Anaconda 다운로드
# https://www.anaconda.com/download

# 가상환경 생성
conda create -n econometrics python=3.10
conda activate econometrics
```

---

## 1.3 필수 라이브러리 소개

### 핵심 라이브러리

#### 1. NumPy
- 수치 계산의 기본
- 배열 연산 및 선형대수
```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
```

#### 2. Pandas
- 데이터 조작 및 분석
- DataFrame 구조로 테이블 데이터 처리
```python
import pandas as pd
df = pd.read_csv('data.csv')
```

---

## 1.3 필수 라이브러리 소개 (계속)

#### 3. Statsmodels
- 통계 모델링 전문 라이브러리
- 회귀분석, 시계열 분석 등
```python
import statsmodels.api as sm
```

#### 4. Matplotlib & Seaborn
- 데이터 시각화
```python
import matplotlib.pyplot as plt
import seaborn as sns
```

#### 5. SciPy
- 과학 계산 및 통계 함수
```python
from scipy import stats
```

---

## 1.4 라이브러리 설치

### 패키지 설치
```bash
# 한 번에 설치하기
conda install numpy pandas statsmodels matplotlib seaborn scipy

# 또는 pip 사용
pip install numpy pandas statsmodels matplotlib seaborn scipy
```

### 설치 확인
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

print("모든 라이브러리 설치 완료!")
```

---

## 1.5 실습 환경 구축

### Jupyter Notebook 시작하기
```bash
# Jupyter Notebook 실행
jupyter notebook

# 또는 Jupyter Lab
jupyter lab
```

### 작업 디렉토리 구조
```
econometrics_course/
├── data/           # 데이터 파일
├── notebooks/      # Jupyter 노트북
├── outputs/        # 결과 및 그래프
└── scripts/        # Python 스크립트
```

---

## 1.6 첫 데이터 분석 실습

### 샘플 데이터 생성
```python
import pandas as pd
import numpy as np

# 간단한 경제 데이터 생성
np.random.seed(42)
data = {
    '소득': np.random.normal(5000, 1000, 100),
    '소비': np.random.normal(3000, 800, 100),
    '저축': np.random.normal(2000, 500, 100)
}
df = pd.DataFrame(data)
```

---

## 1.6 첫 데이터 분석 실습 (계속)

### 기본 탐색
```python
# 데이터 확인
print(df.head())

# 기초 통계량
print(df.describe())

# 데이터 타입 확인
print(df.dtypes)
```

### 간단한 시각화
```python
import matplotlib.pyplot as plt

# 소득과 소비의 관계
plt.scatter(df['소득'], df['소비'])
plt.xlabel('소득')
plt.ylabel('소비')
plt.title('소득-소비 관계')
plt.show()
```

---

## 1.7 첫 회귀분석

### 소득과 소비의 관계 분석
```python
import statsmodels.api as sm

# 독립변수(X)와 종속변수(y)
X = df['소득']
y = df['소비']

# 상수항 추가
X = sm.add_constant(X)

# OLS 회귀분석
model = sm.OLS(y, X)
results = model.fit()

# 결과 출력
print(results.summary())
```

---

## 1.8 결과 해석

### 회귀분석 결과 읽기
- **coef (계수)**: 소득이 1단위 증가할 때 소비의 변화량
- **P>|t| (p-value)**: 통계적 유의성 (보통 0.05 이하)
- **R-squared**: 모델의 설명력 (0~1 사이)

### 예시 해석
```
소득 계수 = 0.75, p-value = 0.001
→ 소득이 1만원 증가하면 소비가 7,500원 증가
→ 통계적으로 유의미함 (p < 0.05)
```

---

## 1.9 Chapter 1 요약

### 배운 내용
✓ 계량경제학의 정의와 Python의 장점  
✓ 필수 라이브러리 설치 및 환경 구축  
✓ Jupyter Notebook 사용법  
✓ 첫 데이터 분석 및 회귀분석 실습

### 다음 단계
- Chapter 2: 경제 데이터 구조 심화 학습
- 실제 데이터 EDA 기법
- 데이터 전처리 마스터하기

---

## 실습 과제

### 과제 1: 환경 구축
- Python 및 필수 라이브러리 설치
- Jupyter Notebook에서 각 라이브러리 import 확인

### 과제 2: 기본 분석
- 제공된 샘플 데이터로 기초통계량 계산
- 간단한 산점도 그리기
- 소득-저축 관계도 회귀분석 해보기

### 과제 3: 탐구
- 온라인에서 간단한 경제 데이터 찾기
- Pandas로 불러와서 head() 와 describe() 실행

---

## 참고 자료

### 공식 문서
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Statsmodels Documentation](https://www.statsmodels.org/)

### 추천 학습 자료
- Python for Data Analysis (Wes McKinney)
- Statsmodels 공식 튜토리얼
- Kaggle Learn: Python 과정

---

## Q&A

질문이 있으시면 편하게 물어보세요!
