# Chapter 4: 경제 데이터의 인과와 해석

---

## 4.1 상관관계 vs 인과관계

### 상관관계 (Correlation)
- 두 변수가 함께 움직이는 관계
- "X와 Y가 관련이 있다"

### 인과관계 (Causation)
- X가 Y를 **직접** 야기하는 관계
- "X가 Y를 발생시킨다"

### 핵심 원칙
**"상관관계가 인과관계를 의미하지 않는다"**
(Correlation does not imply causation)

---

## 4.2 인과관계 예시

### 잘못된 인과 추론
❌ 아이스크림 판매 ↑ → 범죄율 ↑?  
✓ 실제: 기온(제3의 변수)이 둘 다에 영향

❌ 교육 수준 ↑ → 소득 ↑?  
✓ 가능하지만, 능력(관측 불가)이 둘 다에 영향 가능

### 올바른 인과 추론
✓ 최저임금 인상 → 고용에 미치는 영향?  
✓ 교육 프로그램 참여 → 임금 변화?

---

## 4.3 내생성 문제 (Endogeneity)

### 내생성이란?
- 독립변수(X)와 오차항(ε)이 상관관계를 가짐
- OLS 추정량이 편향(biased)되고 일치(inconsistent)하지 않음

### 내생성의 원인

#### 1. 누락변수 편향 (Omitted Variable Bias)
- 중요한 변수를 모델에 포함하지 않음
- 예: 임금 회귀에서 능력 변수 누락

#### 2. 측정오차 (Measurement Error)
- 독립변수가 정확히 측정되지 않음

#### 3. 동시성 (Simultaneity)
- X와 Y가 서로 영향을 주고받음
- 예: 가격과 수요량

---

## 4.4 누락변수 편향 예시

### 문제 상황
```python
# 진짜 모델
임금 = β₀ + β₁×교육 + β₂×능력 + ε

# 추정 모델 (능력 누락)
임금 = β₀ + β₁×교육 + υ
```

### 결과
- 교육과 능력이 양의 상관관계
- β₁ 추정치가 상향 편향 (overestimate)
- 교육의 효과를 과대평가

---

## 4.5 도구변수 (Instrumental Variable)

### 도구변수란?
- 내생성 문제를 해결하는 방법
- 내생 변수를 대신할 수 있는 외생 변수

### 도구변수의 조건

#### 1. 관련성 (Relevance)
- 도구변수(Z)가 내생변수(X)와 상관관계
- Cov(Z, X) ≠ 0

#### 2. 외생성 (Exogeneity)
- 도구변수(Z)가 오차항(ε)과 독립
- Cov(Z, ε) = 0

#### 3. 배제 제약 (Exclusion Restriction)
- Z가 X를 통해서만 Y에 영향
- 직접적인 영향 없음

---

## 4.6 도구변수 예시

### 예시 1: 교육의 수익률
- **내생변수**: 교육년수 (능력과 상관)
- **도구변수**: 출생 분기, 지역별 대학 접근성
- **논리**: 도구변수가 교육에는 영향, 능력과는 무관

### 예시 2: 가격 탄력성
- **내생변수**: 가격 (수요와 동시 결정)
- **도구변수**: 생산비용, 날씨 (공급 측 변수)
- **논리**: 도구변수가 공급에 영향, 수요와는 무관

---

## 4.7 2SLS (Two-Stage Least Squares)

### 2단계 최소자승법

#### Stage 1: 내생변수를 도구변수로 회귀
```
X = π₀ + π₁Z + v
X̂ = fitted values
```

#### Stage 2: 예측값을 사용하여 원래 모델 추정
```
Y = β₀ + β₁X̂ + ε
```

### 직관
- 내생변수 X에서 "좋은 부분"(Z와 관련)만 추출
- 이 부분을 사용하여 Y를 설명

---

## 4.8 Python으로 2SLS 구현

### 데이터 준비
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

np.random.seed(42)
n = 1000

# 도구변수
Z = np.random.normal(0, 1, n)

# 능력 (관측 불가)
ability = np.random.normal(0, 1, n)

# 교육 (도구변수 + 능력에 영향받음)
education = 10 + 2*Z + 1*ability + np.random.normal(0, 1, n)

# 임금 (교육 + 능력에 영향받음)
log_wage = 5 + 0.1*education + 0.5*ability + np.random.normal(0, 0.5, n)

df = pd.DataFrame({
    '도구변수': Z,
    '교육': education,
    '로그임금': log_wage
})
```

---

## 4.9 Python으로 2SLS 구현 (계속)

### OLS vs 2SLS 비교
```python
# 1. OLS (편향된 추정)
X_ols = sm.add_constant(df['교육'])
y = df['로그임금']
ols_model = sm.OLS(y, X_ols).fit()
print("OLS 결과:")
print(ols_model.summary())

# 2. 2SLS (일치 추정)
# Statsmodels IV2SLS
X_exog = np.ones((n, 1))  # 상수항
X_endog = df[['교육']]     # 내생변수
Z_instruments = df[['도구변수']]  # 도구변수

iv_model = IV2SLS(y, X_exog, X_endog, Z_instruments).fit()
print("\n2SLS 결과:")
print(iv_model.summary())
```

---

## 4.10 2SLS 결과 해석

### 계수 비교
```python
print(f"OLS 교육 계수: {ols_model.params['교육']:.4f}")
print(f"2SLS 교육 계수: {iv_model.params['교육']:.4f}")
print(f"진짜 값: 0.1000")
```

### 해석
- OLS: 상향 편향 (능력 효과 포함)
- 2SLS: 진짜 인과효과에 가까움
- 2SLS 표준오차가 더 큼 (trade-off)

---

## 4.11 도구변수 강도 검정

### First-Stage F-test
```python
# Stage 1 회귀
X_stage1 = sm.add_constant(df['도구변수'])
stage1_model = sm.OLS(df['교육'], X_stage1).fit()
print(stage1_model.summary())

# F-statistic 확인
f_stat = stage1_model.fvalue
print(f"\nFirst-stage F-statistic: {f_stat:.2f}")
```

### 판단 기준
- F > 10: 강한 도구변수 (Stock-Yogo)
- F < 10: 약한 도구변수 (weak instrument)
- 약한 도구변수 → 편향, 큰 표준오차

---

## 4.12 시계열 데이터 소개

### 시계열 데이터란?
- 시간 순서대로 관측된 데이터
- 예: GDP, 주가, 실업률, 환율

### 시계열의 특징

#### 1. 추세 (Trend)
- 장기적인 상승/하락 패턴

#### 2. 계절성 (Seasonality)
- 주기적으로 반복되는 패턴
- 예: 분기별, 월별

#### 3. 순환 (Cycle)
- 불규칙한 주기의 변동
- 예: 경기순환

#### 4. 불규칙 변동 (Irregular)
- 예측 불가능한 랜덤 변동

---

## 4.13 시계열 데이터 구조

### Pandas로 시계열 다루기
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 날짜 인덱스 생성
dates = pd.date_range('2010-01-01', periods=120, freq='M')

# 시계열 데이터 생성
np.random.seed(42)
trend = np.linspace(100, 150, 120)
seasonal = 10 * np.sin(np.linspace(0, 10*np.pi, 120))
noise = np.random.normal(0, 5, 120)

ts_data = trend + seasonal + noise

# DataFrame 생성
df_ts = pd.DataFrame({
    'value': ts_data
}, index=dates)

print(df_ts.head())
```

---

## 4.14 시계열 데이터 시각화

### 기본 플롯
```python
# 시계열 플롯
plt.figure(figsize=(12, 6))
plt.plot(df_ts.index, df_ts['value'])
plt.xlabel('날짜')
plt.ylabel('값')
plt.title('시계열 데이터')
plt.grid(True)
plt.show()
```

### 구성요소 분해
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# 시계열 분해
decomposition = seasonal_decompose(df_ts['value'], 
                                   model='additive', 
                                   period=12)

# 플롯
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()
```

---

## 4.15 시계열의 정상성 (Stationarity)

### 정상성이란?
- 평균, 분산, 자기상관이 시간에 따라 일정
- 많은 시계열 모델의 가정

### 정상성의 조건
1. 일정한 평균: E(Yₜ) = μ
2. 일정한 분산: Var(Yₜ) = σ²
3. 공분산이 시차에만 의존: Cov(Yₜ, Yₜ₊ₖ)

### 비정상 시계열
- 추세가 있는 데이터
- 단위근(unit root)을 가진 데이터
- 분산이 변하는 데이터

---

## 4.16 단위근 검정 (Unit Root Test)

### ADF Test (Augmented Dickey-Fuller)
```python
from statsmodels.tsa.stattools import adfuller

# ADF 검정
result = adfuller(df_ts['value'])

print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'\t{key}: {value}')

# 해석
if result[1] < 0.05:
    print("\n결과: 정상 시계열 (단위근 없음)")
else:
    print("\n결과: 비정상 시계열 (단위근 존재)")
```

---

## 4.17 차분 (Differencing)

### 1차 차분
```python
# 차분으로 정상성 만들기
df_ts['diff1'] = df_ts['value'].diff()

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(df_ts.index, df_ts['value'])
axes[0].set_title('원본 데이터')
axes[0].set_ylabel('값')

axes[1].plot(df_ts.index, df_ts['diff1'])
axes[1].set_title('1차 차분')
axes[1].set_ylabel('차분값')
axes[1].axhline(y=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()

# 차분 데이터의 정상성 검정
result_diff = adfuller(df_ts['diff1'].dropna())
print(f"차분 후 p-value: {result_diff[1]:.4f}")
```

---

## 4.18 자기상관 (Autocorrelation)

### ACF (Autocorrelation Function)
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF 플롯
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

plot_acf(df_ts['value'], lags=40, ax=axes[0])
axes[0].set_title('ACF')

plot_pacf(df_ts['value'], lags=40, ax=axes[1])
axes[1].set_title('PACF')

plt.tight_layout()
plt.show()
```

### 해석
- ACF: 시차별 자기상관
- PACF: 편자기상관 (다른 시차 효과 제거)

---

## 4.19 간단한 시계열 모델

### AR(1) 모델 (AutoRegressive)
```python
from statsmodels.tsa.ar_model import AutoReg

# AR(1) 모델
ar_model = AutoReg(df_ts['value'], lags=1).fit()
print(ar_model.summary())

# 예측
forecast = ar_model.predict(start=len(df_ts), end=len(df_ts)+11)
print("\n향후 12개월 예측:")
print(forecast)
```

### AR(1) 형태
```
Yₜ = φ₀ + φ₁Yₜ₋₁ + εₜ
```
- 현재값이 직전 값에 의존

---

## 4.20 패널 데이터 소개

### 패널 데이터란?
- 여러 개체(cross-section)를 시간에 걸쳐 관찰
- 횡단면 + 시계열의 결합

### 패널 데이터 구조
```
개체ID | 시점 | 변수1 | 변수2 | ...
--------|------|-------|-------|-----
   1    | 2020 |  ...  |  ...  | ...
   1    | 2021 |  ...  |  ...  | ...
   2    | 2020 |  ...  |  ...  | ...
   2    | 2021 |  ...  |  ...  | ...
```

---

## 4.21 패널 데이터의 장점

### 1. 더 많은 정보
- 개체 간 + 시간 간 변이 활용
- 더 많은 관측치 → 효율적 추정

### 2. 이질성 통제
- 개체별 고정효과 (관측 불가 특성)
- 시간 고정효과 (공통 충격)

### 3. 동태적 분석
- 시간에 따른 변화 추적
- 인과관계 추론 강화

---

## 4.22 패널 데이터 생성 (Python)

### 샘플 패널 데이터
```python
import pandas as pd
import numpy as np

np.random.seed(42)

# 50개 기업, 10년간
n_firms = 50
n_years = 10

firms = np.repeat(range(1, n_firms+1), n_years)
years = np.tile(range(2014, 2024), n_firms)

# 기업 고정효과 (관측 불가)
firm_effects = np.random.normal(5, 2, n_firms)
firm_effects_expanded = np.repeat(firm_effects, n_years)

# 데이터 생성
df_panel = pd.DataFrame({
    'firm_id': firms,
    'year': years,
    'investment': np.random.normal(100, 20, n_firms*n_years),
    'sales': np.random.normal(1000, 200, n_firms*n_years)
})

# 종속변수 생성 (고정효과 포함)
df_panel['profit'] = (firm_effects_expanded + 
                      0.5 * df_panel['sales'] + 
                      0.3 * df_panel['investment'] + 
                      np.random.normal(0, 50, n_firms*n_years))

print(df_panel.head(15))
```

---

## 4.23 패널 데이터 탐색

### MultiIndex 설정
```python
# MultiIndex 설정
df_panel = df_panel.set_index(['firm_id', 'year'])
print(df_panel.head(10))

# 특정 기업 데이터
firm_1 = df_panel.loc[1]
print("\n기업 1의 시계열:")
print(firm_1)
```

### 기초 통계
```python
# 전체 통계
print(df_panel.describe())

# 기업별 평균
firm_means = df_panel.groupby('firm_id').mean()
print("\n기업별 평균:")
print(firm_means.head())
```

---

## 4.24 패널 회귀모델 유형

### 1. Pooled OLS
- 모든 데이터를 하나로 합쳐 OLS
- 개체 간 이질성 무시
- **문제**: 편향된 추정

### 2. Fixed Effects (FE)
- 개체별 고정효과 포함
- Within estimator
- **장점**: 시간불변 이질성 통제

### 3. Random Effects (RE)
- 고정효과를 랜덤으로 가정
- GLS 추정
- **가정**: 고정효과와 설명변수 독립

---

## 4.25 Python으로 패널 회귀분석

### Linearmodels 라이브러리
```python
# 설치 (필요시)
# !pip install linearmodels

from linearmodels.panel import PanelOLS, RandomEffects
import pandas as pd

# 데이터 준비
df_panel = df_panel.reset_index()
df_panel = df_panel.set_index(['firm_id', 'year'])
```

---

## 4.26 Pooled OLS vs Fixed Effects

### Pooled OLS
```python
# Pooled OLS
pooled_model = PanelOLS.from_formula(
    'profit ~ sales + investment',
    data=df_panel
)
pooled_results = pooled_model.fit()
print("Pooled OLS:")
print(pooled_results)
```

### Fixed Effects
```python
# Fixed Effects (개체 고정효과)
fe_model = PanelOLS.from_formula(
    'profit ~ sales + investment + EntityEffects',
    data=df_panel
)
fe_results = fe_model.fit()
print("\nFixed Effects:")
print(fe_results)
```

---

## 4.27 Random Effects

### Random Effects 모델
```python
# Random Effects
re_model = RandomEffects.from_formula(
    'profit ~ sales + investment',
    data=df_panel
)
re_results = re_model.fit()
print("Random Effects:")
print(re_results)
```

---

## 4.28 Hausman Test

### FE vs RE 선택
```python
# Hausman Test
from linearmodels.panel import compare

# 모델 비교
hausman = compare({
    'Fixed Effects': fe_results,
    'Random Effects': re_results
})
print(hausman)
```

### 해석
- H₀: RE가 일치 추정 (RE 선호)
- H₁: FE가 일치 추정 (FE 선호)
- p < 0.05: FE 사용
- p ≥ 0.05: RE 사용 (더 효율적)

---

## 4.29 시간 고정효과

### 양방향 고정효과 (Two-way FE)
```python
# 개체 + 시간 고정효과
twoway_fe_model = PanelOLS.from_formula(
    'profit ~ sales + investment + EntityEffects + TimeEffects',
    data=df_panel
)
twoway_fe_results = twoway_fe_model.fit()
print("Two-way Fixed Effects:")
print(twoway_fe_results)
```

### 언제 사용?
- 시간에 따른 공통 충격 통제
- 예: 경제위기, 정책 변화, 기술 혁신

---

## 4.30 실습: 패널 데이터 분석

### 실습 데이터
```python
# 국가-연도 패널 데이터 (예시)
np.random.seed(42)
n_countries = 30
n_years = 20

countries = np.repeat(range(1, n_countries+1), n_years)
years = np.tile(range(2004, 2024), n_countries)

# 국가 고정효과
country_fe = np.random.normal(50, 10, n_countries)
country_fe_exp = np.repeat(country_fe, n_years)

df_country = pd.DataFrame({
    'country_id': countries,
    'year': years,
    'gdp_growth': np.random.normal(3, 2, n_countries*n_years),
    'investment_rate': np.random.uniform(15, 35, n_countries*n_years),
    'education': np.random.normal(10, 2, n_countries*n_years)
})

# 1인당 GDP 생성
df_country['gdp_per_capita'] = (country_fe_exp + 
                                 5 * df_country['education'] + 
                                 2 * df_country['investment_rate'] + 
                                 np.random.normal(0, 10, n_countries*n_years))

df_country = df_country.set_index(['country_id', 'year'])
```

---

## 4.31 실습: 모델 비교

### 세 가지 모델 추정
```python
# 1. Pooled OLS
pooled = PanelOLS.from_formula(
    'gdp_per_capita ~ education + investment_rate',
    data=df_country
).fit()

# 2. Fixed Effects
fe = PanelOLS.from_formula(
    'gdp_per_capita ~ education + investment_rate + EntityEffects',
    data=df_country
).fit()

# 3. Random Effects
re = RandomEffects.from_formula(
    'gdp_per_capita ~ education + investment_rate',
    data=df_country
).fit()

# 결과 비교
print("계수 비교:")
print(f"Pooled - education: {pooled.params['education']:.4f}")
print(f"FE - education: {fe.params['education']:.4f}")
print(f"RE - education: {re.params['education']:.4f}")
```

---

## 4.32 Chapter 4 요약

### 배운 내용
✓ 인과관계 vs 상관관계  
✓ 내생성 문제와 원인  
✓ 도구변수(IV)와 2SLS  
✓ 시계열 데이터 기초 (정상성, 차분, 자기상관)  
✓ 패널 데이터 구조와 분석 방법  
✓ Pooled OLS, Fixed Effects, Random Effects  
✓ Hausman Test

### 핵심 포인트
- 내생성 → 편향된 추정
- 도구변수로 인과효과 추정
- 패널 데이터로 이질성 통제

---

## 4.33 실습 과제

### 과제 1: 2SLS 분석
- 제공된 데이터로 내생성 확인
- 적절한 도구변수 찾기
- OLS vs 2SLS 결과 비교

### 과제 2: 시계열 분석
- 실제 경제 시계열 데이터 다운로드 (예: GDP, 실업률)
- 정상성 검정 (ADF test)
- 차분 및 ACF/PACF 분석

### 과제 3: 패널 데이터 분석
- 패널 데이터로 세 가지 모델 추정
- Hausman test 수행
- 결과 해석 및 비교

---

## 4.34 다음 단계

### Chapter 5 예고: 실제 경제 데이터로 본 실전 분석
- 실제 공개 경제 데이터셋 활용
- 사례 1: 임금 결정요인 실증분석
- 사례 2: 주택가격 결정요인 분석
- 종합 프로젝트

### 준비사항
- Chapter 1-4 복습
- 실제 데이터 출처 탐색
- 연구 질문 생각해보기

---

## 참고 자료

### 교재
- "Econometric Analysis of Cross Section and Panel Data" by Wooldridge
- "Introduction to Econometrics" by Stock and Watson

### Python 라이브러리
- `linearmodels`: 패널 데이터 분석
- `statsmodels`: 시계열, IV 분석
- `pandas`: 데이터 처리

### 데이터 출처
- World Bank Open Data
- OECD Statistics
- Penn World Table

---

## Q&A

질문이 있으시면 편하게 물어보세요!
