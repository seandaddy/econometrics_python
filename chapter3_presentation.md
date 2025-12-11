# Chapter 3: 회귀분석의 모든 것

---

## 3.1 회귀분석이란?

### 정의
- 변수 간의 관계를 수학적으로 모델링
- 독립변수(X)가 종속변수(Y)에 미치는 영향 분석
- 예측과 인과관계 추론에 활용

### 경제학에서의 활용
- 소득이 소비에 미치는 영향
- 교육이 임금에 미치는 효과
- 금리가 투자에 미치는 영향

---

## 3.2 단순 선형회귀

### 모델 형태
```
Y = β₀ + β₁X + ε
```

- Y: 종속변수 (설명하려는 변수)
- X: 독립변수 (설명변수)
- β₀: 절편 (intercept)
- β₁: 기울기 (slope)
- ε: 오차항 (error term)

### 예시
```
소비 = β₀ + β₁ × 소득 + ε
```

---

## 3.3 OLS (최소자승법)

### OLS의 원리
- Ordinary Least Squares
- 잔차 제곱합을 최소화하는 β₀, β₁ 추정
- 잔차: 실제값 - 예측값

### 수식
```
최소화: Σ(yᵢ - ŷᵢ)²
```

### OLS 추정량
```
β₁ = Cov(X,Y) / Var(X)
β₀ = Ȳ - β₁X̄
```

---

## 3.4 Python으로 단순 회귀분석

### 데이터 준비
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 샘플 데이터
np.random.seed(42)
n = 100
소득 = np.random.normal(5000, 1000, n)
소비 = 2000 + 0.6 * 소득 + np.random.normal(0, 500, n)

df = pd.DataFrame({'소득': 소득, '소비': 소비})
```

---

## 3.5 Python으로 단순 회귀분석 (계속)

### 회귀분석 실행
```python
# 독립변수와 종속변수
X = df['소득']
y = df['소비']

# 상수항 추가
X = sm.add_constant(X)

# OLS 모델 적합
model = sm.OLS(y, X)
results = model.fit()

# 결과 출력
print(results.summary())
```

---

## 3.6 회귀분석 결과 해석

### 주요 통계량

#### 1. 계수 (Coefficients)
- **const (β₀)**: 소득이 0일 때 예상 소비
- **소득 (β₁)**: 소득이 1단위 증가할 때 소비의 변화량

#### 2. p-value (P>|t|)
- 통계적 유의성 검정
- p < 0.05: 통계적으로 유의
- p < 0.01: 매우 유의

#### 3. R-squared (결정계수)
- 모델의 설명력 (0~1)
- 독립변수가 종속변수 분산의 몇 %를 설명하는가

---

## 3.7 회귀 결과 시각화

### 산점도와 회귀선
```python
# 산점도
plt.scatter(df['소득'], df['소비'], alpha=0.5)

# 회귀선
X_plot = np.linspace(df['소득'].min(), df['소득'].max(), 100)
X_plot_const = sm.add_constant(X_plot)
y_plot = results.predict(X_plot_const)

plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='회귀선')
plt.xlabel('소득')
plt.ylabel('소비')
plt.title('소득-소비 관계')
plt.legend()
plt.show()
```

---

## 3.8 다중 선형회귀

### 모델 형태
```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε
```

### 예시: 소비 함수
```
소비 = β₀ + β₁×소득 + β₂×자산 + β₃×가구원수 + ε
```

### 다중회귀의 장점
- 여러 요인의 영향을 동시에 고려
- 누락변수 편향(omitted variable bias) 완화
- 더 정확한 예측

---

## 3.9 Python으로 다중 회귀분석

### 데이터 준비
```python
# 다변수 데이터 생성
np.random.seed(42)
n = 200

data = {
    '소득': np.random.normal(5000, 1000, n),
    '자산': np.random.normal(10000, 3000, n),
    '가구원수': np.random.choice([1,2,3,4,5], n),
    '연령': np.random.normal(40, 10, n)
}

df = pd.DataFrame(data)

# 소비 생성 (여러 변수의 선형결합)
df['소비'] = (1000 + 
              0.5 * df['소득'] + 
              0.2 * df['자산'] + 
              300 * df['가구원수'] + 
              10 * df['연령'] + 
              np.random.normal(0, 500, n))
```

---

## 3.10 Python으로 다중 회귀분석 (계속)

### 회귀분석 실행
```python
# 독립변수들
X = df[['소득', '자산', '가구원수', '연령']]
y = df['소비']

# 상수항 추가
X = sm.add_constant(X)

# OLS 모델
model = sm.OLS(y, X)
results = model.fit()

# 결과
print(results.summary())
```

---

## 3.11 다중회귀 결과 해석

### 계수 해석 (Ceteris Paribus)
- "다른 조건이 동일할 때" (other things being equal)
- β₁: 자산, 가구원수, 연령이 고정일 때, 소득 1단위 증가에 따른 소비 변화

### 예시 해석
```
소득 계수 = 0.48, p = 0.000
→ 다른 조건이 같을 때, 소득이 1만원 증가하면 
  소비가 4,800원 증가
→ 통계적으로 유의미 (p < 0.05)
```

---

## 3.12 회귀진단 (1): 잔차분석

### 잔차(Residuals)란?
```
잔차 = 실제값 - 예측값
eᵢ = yᵢ - ŷᵢ
```

### OLS 가정
1. **선형성**: X와 Y의 관계가 선형
2. **독립성**: 관측치들이 서로 독립
3. **등분산성**: 잔차의 분산이 일정 (homoscedasticity)
4. **정규성**: 잔차가 정규분포
5. **외생성**: X와 ε가 독립

---

## 3.13 잔차 플롯

### 잔차 vs 적합값
```python
# 잔차 계산
fitted_values = results.fittedvalues
residuals = results.resid

# 잔차 플롯
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('적합값')
plt.ylabel('잔차')
plt.title('잔차 플롯')
plt.show()
```

### 해석
- 패턴이 없어야 함 (random scatter)
- 패턴 존재 → 선형성 또는 등분산성 위배

---

## 3.14 잔차의 정규성 검정

### Q-Q Plot
```python
from scipy import stats
import matplotlib.pyplot as plt

# Q-Q plot
fig = sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot')
plt.show()
```

### 히스토그램
```python
# 잔차 히스토그램
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('잔차')
plt.ylabel('빈도')
plt.title('잔차 분포')
plt.show()
```

### 해석
- 점들이 45도 선에 가까울수록 정규분포
- 크게 벗어나면 정규성 위배

---

## 3.15 등분산성 검정

### Breusch-Pagan Test
```python
from statsmodels.stats.diagnostic import het_breuschpagan

# 등분산성 검정
bp_test = het_breuschpagan(residuals, X)
labels = ['LM Statistic', 'LM-Test p-value', 
          'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))
```

### 해석
- H₀: 등분산성 만족
- p-value > 0.05: 등분산성 만족
- p-value < 0.05: 이분산성(heteroscedasticity) 존재

---

## 3.16 이분산성 처리

### 1. 강건 표준오차 (Robust Standard Errors)
```python
# White's robust standard errors
results_robust = model.fit(cov_type='HC3')
print(results_robust.summary())
```

### 2. 가중최소자승법 (WLS)
```python
from statsmodels.regression.linear_model import WLS

# 가중치 계산 (잔차의 역수)
weights = 1 / (residuals**2)

# WLS 모델
wls_model = WLS(y, X, weights=weights)
wls_results = wls_model.fit()
print(wls_results.summary())
```

---

## 3.17 회귀진단 (2): 다중공선성

### 다중공선성(Multicollinearity)이란?
- 독립변수들 간의 높은 상관관계
- 계수 추정의 불안정성 초래
- 표준오차 증가

### 문제점
- 통계적 유의성 감소
- 계수 해석 어려움
- 예측력은 유지되지만 해석이 불안정

---

## 3.18 VIF (Variance Inflation Factor)

### VIF 계산
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF 계산
X_for_vif = df[['소득', '자산', '가구원수', '연령']]
vif_data = pd.DataFrame()
vif_data['Variable'] = X_for_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_for_vif.values, i) 
                   for i in range(X_for_vif.shape[1])]
print(vif_data)
```

### VIF 해석
- VIF < 5: 문제 없음
- 5 ≤ VIF < 10: 주의 필요
- VIF ≥ 10: 심각한 다중공선성

---

## 3.19 다중공선성 해결 방법

### 1. 상관관계가 높은 변수 제거
```python
# 상관계수 확인
corr_matrix = X_for_vif.corr()
print(corr_matrix)

# 높은 상관관계 변수 중 하나 제거
X_reduced = df[['소득', '가구원수', '연령']]
```

### 2. 변수 결합
```python
# 두 변수를 하나로 결합 (예: 주성분분석)
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
df['소득자산종합'] = pca.fit_transform(df[['소득', '자산']])
```

### 3. Ridge/Lasso 회귀 (Chapter 7에서 다룸)

---

## 3.20 회귀진단 (3): 영향력 있는 관측치

### Leverage (지렛대값)
- X 공간에서 멀리 떨어진 관측치
- 회귀선에 큰 영향

### Cook's Distance
- 특정 관측치가 회귀 결과에 미치는 영향
- Cook's D > 1: 영향력 있는 관측치

```python
from statsmodels.stats.outliers_influence import OLSInfluence

influence = OLSInfluence(results)
cooks_d = influence.cooks_distance[0]

# Cook's Distance 시각화
plt.stem(range(len(cooks_d)), cooks_d)
plt.axhline(y=1, color='r', linestyle='--', label='Threshold=1')
plt.xlabel('관측치 인덱스')
plt.ylabel("Cook's Distance")
plt.legend()
plt.show()
```

---

## 3.21 실습: 임금 결정요인 분석

### 데이터 생성
```python
np.random.seed(42)
n = 500

# 임금 데이터
df_wage = pd.DataFrame({
    '교육년수': np.random.normal(14, 3, n),
    '경력년수': np.random.uniform(0, 30, n),
    '성별': np.random.choice([0, 1], n),  # 0: 여성, 1: 남성
    '지역': np.random.choice([0, 1], n)   # 0: 지방, 1: 수도권
})

# 임금 생성 (로그 임금)
df_wage['log_임금'] = (8 + 
                       0.1 * df_wage['교육년수'] + 
                       0.03 * df_wage['경력년수'] + 
                       0.2 * df_wage['성별'] + 
                       0.15 * df_wage['지역'] + 
                       np.random.normal(0, 0.3, n))

# 실제 임금 (원 단위)
df_wage['임금'] = np.exp(df_wage['log_임금']) * 1000
```

---

## 3.22 실습: 임금 회귀분석

### 모델 1: 단순 회귀
```python
# 교육년수만
X1 = sm.add_constant(df_wage['교육년수'])
y = df_wage['log_임금']

model1 = sm.OLS(y, X1).fit()
print(model1.summary())
```

### 모델 2: 다중 회귀
```python
# 모든 변수 포함
X2 = df_wage[['교육년수', '경력년수', '성별', '지역']]
X2 = sm.add_constant(X2)

model2 = sm.OLS(y, X2).fit()
print(model2.summary())
```

---

## 3.23 실습: 모델 비교

### 결정계수 비교
```python
print(f"모델1 R²: {model1.rsquared:.4f}")
print(f"모델2 R²: {model2.rsquared:.4f}")
```

### 조정된 결정계수 (Adjusted R²)
- 변수 개수를 고려한 R²
- 불필요한 변수 추가를 패널티

```python
print(f"모델1 Adj R²: {model1.rsquared_adj:.4f}")
print(f"모델2 Adj R²: {model2.rsquared_adj:.4f}")
```

---

## 3.24 실습: 회귀진단

### 잔차 플롯
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 잔차 vs 적합값
axes[0, 0].scatter(model2.fittedvalues, model2.resid, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('적합값')
axes[0, 0].set_ylabel('잔차')
axes[0, 0].set_title('잔차 플롯')

# 2. Q-Q plot
sm.qqplot(model2.resid, line='45', ax=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# 3. 잔차 히스토그램
axes[1, 0].hist(model2.resid, bins=30, edgecolor='black')
axes[1, 0].set_xlabel('잔차')
axes[1, 0].set_title('잔차 분포')

# 4. Scale-Location plot
axes[1, 1].scatter(model2.fittedvalues, np.sqrt(np.abs(model2.resid)), alpha=0.5)
axes[1, 1].set_xlabel('적합값')
axes[1, 1].set_ylabel('√|잔차|')
axes[1, 1].set_title('Scale-Location Plot')

plt.tight_layout()
plt.show()
```

---

## 3.25 실습: VIF 확인

### 다중공선성 진단
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = df_wage[['교육년수', '경력년수', '성별', '지역']]

vif_data = pd.DataFrame()
vif_data['Variable'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) 
                   for i in range(X_vif.shape[1])]

print(vif_data)
```

---

## 3.26 로그 변환과 해석

### 로그-로그 모델
```
log(Y) = β₀ + β₁ log(X) + ε
β₁: 탄력성 (X가 1% 증가 → Y가 β₁% 증가)
```

### 로그-선형 모델
```
log(Y) = β₀ + β₁ X + ε
β₁ × 100: X가 1단위 증가 → Y가 (β₁×100)% 증가
```

### 선형-로그 모델
```
Y = β₀ + β₁ log(X) + ε
β₁: X가 1% 증가 → Y가 β₁/100 단위 증가
```

---

## 3.27 실습: 로그 변환

### 로그-선형 모델 (우리 예시)
```python
# log_임금 = β₀ + β₁×교육년수 + ...
# 해석: 교육년수가 1년 증가 → 임금이 (β₁×100)% 증가

print(f"교육년수 계수: {model2.params['교육년수']:.4f}")
print(f"해석: 교육년수 1년 증가 → 임금 {model2.params['교육년수']*100:.2f}% 증가")
```

### 더미변수 해석
```python
# 성별 계수
gender_coef = model2.params['성별']
print(f"성별 계수: {gender_coef:.4f}")
print(f"해석: 남성이 여성보다 {(np.exp(gender_coef)-1)*100:.2f}% 임금 높음")
```

---

## 3.28 예측하기

### 새로운 데이터로 예측
```python
# 새로운 관측치
new_data = pd.DataFrame({
    'const': [1],
    '교육년수': [16],
    '경력년수': [5],
    '성별': [1],
    '지역': [1]
})

# 예측 (로그 임금)
log_wage_pred = model2.predict(new_data)
print(f"예측 로그 임금: {log_wage_pred[0]:.4f}")

# 실제 임금으로 변환
wage_pred = np.exp(log_wage_pred[0]) * 1000
print(f"예측 임금: {wage_pred:.0f}원")
```

---

## 3.29 신뢰구간과 예측구간

### 신뢰구간 (Confidence Interval)
- 평균 예측값의 불확실성
```python
predictions = model2.get_prediction(new_data)
pred_summary = predictions.summary_frame(alpha=0.05)

print(pred_summary)
# mean: 예측값
# mean_ci_lower, mean_ci_upper: 95% 신뢰구간
```

### 예측구간 (Prediction Interval)
- 개별 관측치의 불확실성 (더 넓음)
- obs_ci_lower, obs_ci_upper

---

## 3.30 Chapter 3 요약

### 배운 내용
✓ 단순/다중 선형회귀 이론  
✓ OLS 추정 방법  
✓ Statsmodels로 회귀분석 실습  
✓ 회귀 결과 해석 (계수, p-value, R²)  
✓ 회귀진단: 잔차분석, 다중공선성, 영향력 관측치  
✓ 로그 변환과 해석  
✓ 예측 및 신뢰구간

---

## 3.31 실습 과제

### 과제 1: 기본 회귀분석
- 제공된 데이터로 다중회귀분석 수행
- 모든 계수 해석하기
- R²와 Adjusted R² 비교

### 과제 2: 회귀진단
- 잔차 플롯 그리기 및 해석
- VIF 계산하여 다중공선성 확인
- Cook's Distance로 영향력 관측치 탐지

### 과제 3: 모델 개선
- 이상치 제거 후 재분석
- 변수 변환 (로그 등) 시도
- 모델 성능 비교

---

## 3.32 다음 단계

### Chapter 4 예고: 경제 데이터의 인과와 해석
- 인과관계 vs 상관관계
- 내생성 문제와 해결책
- 도구변수(IV)와 2SLS
- 시계열 데이터 기초
- 패널 데이터 분석

### 준비사항
- 회귀분석 개념 복습
- 내생성(endogeneity) 개념 예습
- 시계열/패널 데이터 구조 이해

---

## 참고 자료

### 교재
- "Introductory Econometrics" by Wooldridge
- "Econometric Analysis" by Greene

### 온라인 리소스
- Statsmodels Documentation
- Cross Validated (stats.stackexchange.com)

### Python 라이브러리
- Statsmodels: 통계 모델링
- Scikit-learn: 머신러닝 회귀
- Scipy: 통계 검정

---

## Q&A

질문이 있으시면 편하게 물어보세요!
