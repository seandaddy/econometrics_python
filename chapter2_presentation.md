# Chapter 2: 경제 데이터 이해 및 분석 기초

---

## 2.1 경제 데이터의 특징

### 경제 데이터의 유형

#### 1. 횡단면 데이터 (Cross-sectional Data)
- 특정 시점의 여러 개체 관찰
- 예: 2023년 가구별 소득 데이터

#### 2. 시계열 데이터 (Time Series Data)
- 하나의 개체를 시간에 따라 관찰
- 예: 한국의 월별 실업률 (2010-2023)

#### 3. 패널 데이터 (Panel Data)
- 여러 개체를 시간에 따라 관찰
- 예: 100개 기업의 5년간 매출 데이터

---

## 2.2 경제 데이터의 구조

### 데이터 형식

#### CSV (Comma-Separated Values)
- 가장 흔한 형식
- 텍스트 기반, 호환성 높음

#### Excel (.xlsx, .xls)
- 비즈니스에서 많이 사용
- 여러 시트 지원

#### 기타 형식
- JSON: 웹 API 데이터
- SQL Database: 대용량 데이터
- Stata (.dta), SAS, SPSS: 통계 소프트웨어

---

## 2.3 Pandas로 데이터 불러오기

### CSV 파일 읽기
```python
import pandas as pd

# 기본 읽기
df = pd.read_csv('economic_data.csv')

# 옵션 지정
df = pd.read_csv('economic_data.csv',
                 encoding='utf-8',
                 sep=',',
                 index_col=0)
```

### Excel 파일 읽기
```python
# Excel 파일
df = pd.read_excel('economic_data.xlsx', 
                   sheet_name='Sheet1')

# 여러 시트 읽기
dfs = pd.read_excel('data.xlsx', 
                    sheet_name=None)  # 모든 시트
```

---

## 2.3 Pandas로 데이터 불러오기 (계속)

### 기타 형식
```python
# Stata 파일
df = pd.read_stata('data.dta')

# JSON 파일
df = pd.read_json('data.json')

# SQL Database
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table', conn)
```

---

## 2.4 EDA (탐색적 데이터 분석) 핵심

### 데이터 첫 확인

```python
# 처음 5행 확인
df.head()

# 마지막 5행 확인
df.tail()

# 데이터 형태 (행, 열)
df.shape

# 컬럼명과 데이터 타입
df.info()

# 컬럼 목록
df.columns
```

---

## 2.5 기초 통계량 확인

### describe() 메서드
```python
# 수치형 변수의 기초통계
df.describe()

# 모든 변수 포함
df.describe(include='all')
```

### 개별 통계량
```python
# 평균
df['소득'].mean()

# 중앙값
df['소득'].median()

# 표준편차
df['소득'].std()

# 최소값, 최대값
df['소득'].min(), df['소득'].max()
```

---

## 2.6 데이터 분포 확인

### 히스토그램
```python
import matplotlib.pyplot as plt

df['소득'].hist(bins=30)
plt.xlabel('소득')
plt.ylabel('빈도')
plt.title('소득 분포')
plt.show()
```

### 박스플롯
```python
df.boxplot(column='소득')
plt.ylabel('소득')
plt.title('소득 박스플롯')
plt.show()
```

---

## 2.7 변수 간 관계 탐색

### 산점도
```python
# 기본 산점도
plt.scatter(df['소득'], df['소비'])
plt.xlabel('소득')
plt.ylabel('소비')
plt.show()
```

### 상관계수
```python
# 상관계수 행렬
df.corr()

# 특정 변수 간 상관계수
df['소득'].corr(df['소비'])
```

---

## 2.8 Seaborn을 활용한 시각화

### 상관계수 히트맵
```python
import seaborn as sns

# 히트맵
sns.heatmap(df.corr(), 
            annot=True,      # 숫자 표시
            cmap='coolwarm', # 색상
            center=0)
plt.title('상관계수 행렬')
plt.show()
```

### Pairplot
```python
# 모든 변수 쌍의 관계 시각화
sns.pairplot(df)
plt.show()

# 특정 변수만
sns.pairplot(df[['소득', '소비', '저축']])
plt.show()
```

---

## 2.9 결측치 (Missing Values) 이해

### 결측치 확인
```python
# 결측치 개수
df.isnull().sum()

# 결측치 비율
df.isnull().mean() * 100

# 결측치 시각화
import seaborn as sns
sns.heatmap(df.isnull(), 
            cbar=False,
            yticklabels=False)
plt.show()
```

---

## 2.10 결측치 처리 방법

### 1. 삭제 (Deletion)
```python
# 결측치가 있는 행 삭제
df_clean = df.dropna()

# 특정 열의 결측치만 삭제
df_clean = df.dropna(subset=['소득'])

# 모든 값이 결측인 행만 삭제
df_clean = df.dropna(how='all')
```

### 주의사항
- 데이터 손실 발생
- 표본 크기 감소
- 편향(bias) 발생 가능

---

## 2.11 결측치 대체 (Imputation)

### 2. 평균/중앙값/최빈값으로 대체
```python
# 평균으로 대체
df['소득'].fillna(df['소득'].mean(), inplace=True)

# 중앙값으로 대체
df['소득'].fillna(df['소득'].median(), inplace=True)

# 최빈값으로 대체
df['지역'].fillna(df['지역'].mode()[0], inplace=True)
```

### 3. 전방/후방 채우기 (시계열 데이터)
```python
# 이전 값으로 채우기
df.fillna(method='ffill', inplace=True)

# 다음 값으로 채우기
df.fillna(method='bfill', inplace=True)
```

---

## 2.12 이상치 (Outliers) 이해

### 이상치란?
- 다른 관측값과 크게 다른 값
- 데이터 입력 오류 또는 실제 극단값
- 분석 결과에 큰 영향을 미칠 수 있음

### 이상치 탐지 방법

#### 1. 시각적 방법
```python
# 박스플롯으로 확인
df.boxplot(column='소득')
plt.show()
```

---

## 2.13 이상치 탐지: 통계적 방법

### 2. IQR (Interquartile Range) 방법
```python
# IQR 계산
Q1 = df['소득'].quantile(0.25)
Q3 = df['소득'].quantile(0.75)
IQR = Q3 - Q1

# 이상치 범위 정의
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 확인
outliers = df[(df['소득'] < lower_bound) | 
              (df['소득'] > upper_bound)]
print(f"이상치 개수: {len(outliers)}")
```

---

## 2.14 이상치 탐지: Z-score

### 3. Z-score 방법
```python
from scipy import stats

# Z-score 계산
z_scores = np.abs(stats.zscore(df['소득']))

# 임계값 설정 (보통 3)
threshold = 3
outliers = df[z_scores > threshold]
print(f"이상치 개수: {len(outliers)}")
```

### Z-score 해석
- |Z| > 3: 이상치로 간주
- 평균에서 3 표준편차 이상 떨어진 값

---

## 2.15 이상치 처리 방법

### 1. 삭제
```python
# IQR 기반 이상치 제거
df_clean = df[(df['소득'] >= lower_bound) & 
              (df['소득'] <= upper_bound)]
```

### 2. 변환 (Transformation)
```python
# 로그 변환
df['소득_log'] = np.log(df['소득'])

# 제곱근 변환
df['소득_sqrt'] = np.sqrt(df['소득'])
```

### 3. Winsorization (극단값 제한)
```python
from scipy.stats.mstats import winsorize

# 상하위 5%를 해당 백분위수 값으로 대체
df['소득_win'] = winsorize(df['소득'], 
                           limits=[0.05, 0.05])
```

---

## 2.16 실습: 경제 데이터 EDA

### 실습 데이터 생성
```python
import pandas as pd
import numpy as np

np.random.seed(42)

# 1000명의 가구 데이터 생성
n = 1000
data = {
    '가구ID': range(1, n+1),
    '소득': np.random.lognormal(10, 0.5, n),
    '소비': np.random.lognormal(9, 0.6, n),
    '가구원수': np.random.choice([1,2,3,4,5], n),
    '지역': np.random.choice(['서울','경기','부산','기타'], n),
    '교육년수': np.random.normal(14, 3, n)
}

df = pd.DataFrame(data)

# 일부 결측치 생성
df.loc[np.random.choice(df.index, 50), '소득'] = np.nan
df.loc[np.random.choice(df.index, 30), '교육년수'] = np.nan
```

---

## 2.17 실습: 데이터 탐색

### Step 1: 기본 확인
```python
# 데이터 구조
print(df.shape)
print(df.info())

# 처음 몇 행
print(df.head(10))

# 기초 통계
print(df.describe())
```

---

## 2.18 실습: 결측치 처리

### Step 2: 결측치 확인 및 처리
```python
# 결측치 확인
print(df.isnull().sum())

# 결측치 비율
print(df.isnull().mean() * 100)

# 소득: 중앙값으로 대체
df['소득'].fillna(df['소득'].median(), inplace=True)

# 교육년수: 평균으로 대체
df['교육년수'].fillna(df['교육년수'].mean(), inplace=True)

# 확인
print(df.isnull().sum())
```

---

## 2.19 실습: 이상치 처리

### Step 3: 이상치 탐지 및 처리
```python
# 소득 이상치 확인
Q1 = df['소득'].quantile(0.25)
Q3 = df['소득'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 이상치 개수
outliers = df[(df['소득'] < lower) | (df['소득'] > upper)]
print(f"이상치 개수: {len(outliers)}")

# 이상치 제거
df_clean = df[(df['소득'] >= lower) & (df['소득'] <= upper)]
print(f"정제 후 데이터: {df_clean.shape}")
```

---

## 2.20 실습: 시각화

### Step 4: 데이터 시각화
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 소득 분포
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_clean['소득'], bins=30)
axes[0].set_title('소득 분포')
axes[0].set_xlabel('소득')

axes[1].boxplot(df_clean['소득'])
axes[1].set_title('소득 박스플롯')
axes[1].set_ylabel('소득')

plt.tight_layout()
plt.show()
```

---

## 2.21 실습: 상관관계 분석

### Step 5: 변수 간 관계
```python
# 수치형 변수만 선택
numeric_cols = ['소득', '소비', '가구원수', '교육년수']
df_numeric = df_clean[numeric_cols]

# 상관계수 행렬
corr_matrix = df_numeric.corr()
print(corr_matrix)

# 히트맵
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('상관계수 행렬')
plt.show()
```

---

## 2.22 실습: 그룹별 분석

### Step 6: 범주형 변수 분석
```python
# 지역별 평균 소득
region_income = df_clean.groupby('지역')['소득'].mean()
print(region_income)

# 시각화
region_income.plot(kind='bar')
plt.title('지역별 평균 소득')
plt.xlabel('지역')
plt.ylabel('평균 소득')
plt.show()

# 가구원수별 평균 소비
household_consumption = df_clean.groupby('가구원수')['소비'].mean()
print(household_consumption)
```

---

## 2.23 데이터 변환 기법

### 로그 변환
```python
# 소득의 분포가 치우쳐있을 때
df_clean['소득_log'] = np.log(df_clean['소득'])

# 변환 전후 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_clean['소득'], bins=30)
axes[0].set_title('원본 소득')

axes[1].hist(df_clean['소득_log'], bins=30)
axes[1].set_title('로그 변환 소득')

plt.tight_layout()
plt.show()
```

---

## 2.24 데이터 정규화/표준화

### 표준화 (Standardization)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_clean['소득_표준화'] = scaler.fit_transform(
    df_clean[['소득']]
)

# 평균 0, 표준편차 1
print(df_clean['소득_표준화'].mean())  # ~0
print(df_clean['소득_표준화'].std())   # ~1
```

### 정규화 (Normalization)
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_clean['소득_정규화'] = scaler.fit_transform(
    df_clean[['소득']]
)

# 0과 1 사이 값
print(df_clean['소득_정규화'].min())  # 0
print(df_clean['소득_정규화'].max())  # 1
```

---

## 2.25 Chapter 2 요약

### 배운 내용
✓ 경제 데이터의 유형 (횡단면, 시계열, 패널)  
✓ Pandas로 다양한 형식의 데이터 불러오기  
✓ EDA 핵심 기법 (describe, 시각화, 상관분석)  
✓ 결측치 탐지 및 처리 방법  
✓ 이상치 탐지 및 처리 방법  
✓ 데이터 변환 기법

### 핵심 포인트
- 데이터 분석 전 반드시 EDA 수행
- 결측치와 이상치는 맥락에 맞게 처리
- 시각화는 데이터 이해의 핵심 도구

---

## 2.26 실습 과제

### 과제 1: 실제 데이터 EDA
- 공공데이터포털 또는 KOSIS에서 경제 데이터 다운로드
- Pandas로 불러오기
- 기초통계량 및 시각화 수행

### 과제 2: 결측치/이상치 처리
- 제공된 데이터에서 결측치 비율 계산
- 적절한 방법으로 결측치 처리
- IQR 방법으로 이상치 탐지 및 처리

### 과제 3: 상관분석
- 수치형 변수 간 상관계수 계산
- 히트맵으로 시각화
- 가장 강한 상관관계를 보이는 변수 쌍 찾기

---

## 2.27 다음 단계

### Chapter 3 예고: 회귀분석의 모든 것
- 단순/다중 선형회귀 이론
- OLS 추정과 해석
- Statsmodels 실전 활용
- 회귀진단 (잔차분석, 다중공선성 등)

### 준비사항
- Chapter 2의 데이터 전처리 기법 복습
- 선형대수 기초 (행렬, 벡터)
- 통계학 기초 (평균, 분산, 상관계수)

---

## 참고 자료

### 데이터 출처
- [KOSIS (국가통계포털)](https://kosis.kr/)
- [공공데이터포털](https://www.data.go.kr/)
- [한국은행 경제통계시스템](https://ecos.bok.or.kr/)
- [OECD Data](https://data.oecd.org/)

### 학습 자료
- Pandas User Guide
- Seaborn Tutorial
- "Python for Data Analysis" by Wes McKinney

---

## Q&A

질문이 있으시면 편하게 물어보세요!
