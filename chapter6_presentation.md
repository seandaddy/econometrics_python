# Chapter 6: 모델 선택과 심화 분석

---

## 6.1 모델 선택의 중요성

### 왜 모델 선택이 중요한가?

#### 1. 편향-분산 트레이드오프
- **단순한 모델**: 낮은 분산, 높은 편향 (underfitting)
- **복잡한 모델**: 낮은 편향, 높은 분산 (overfitting)
- **목표**: 최적의 균형점 찾기

#### 2. 해석 가능성 vs 예측력
- **선형 모델**: 해석 쉬움, 예측력 제한적
- **비선형 모델**: 예측력 높음, 해석 어려움

#### 3. 데이터 구조에 따른 선택
- **횡단면**: OLS, Logit/Probit
- **시계열**: ARIMA, VAR
- **패널**: Fixed Effects, Random Effects

---

## 6.2 모델 비교 프레임워크

### 모델 평가 기준

#### 1. 적합도 (Goodness of Fit)
- R², Adjusted R²
- AIC, BIC

#### 2. 예측력 (Predictive Power)
- MSE, RMSE, MAE
- Out-of-sample performance

#### 3. 경제적 의미
- 계수의 부호와 크기
- 이론과의 일치성

#### 4. 통계적 유의성
- p-values
- 신뢰구간

---

## 6.3 선형 vs 비선형 모델

### 선형 모델
```python
# 선형 회귀
Y = β₀ + β₁X₁ + β₂X₂ + ε
```

**장점**
- 해석 간단
- 추정 빠름
- 이론적 근거 명확

**단점**
- 선형 관계 가정
- 상호작용 효과 제한적

---

## 6.4 비선형 모델 유형

### 1. 다항 회귀 (Polynomial Regression)
```python
Y = β₀ + β₁X + β₂X² + β₃X³ + ε
```

### 2. 로그 변환
```python
log(Y) = β₀ + β₁log(X) + ε
```

### 3. 상호작용 항
```python
Y = β₀ + β₁X₁ + β₂X₂ + β₃(X₁×X₂) + ε
```

### 4. 스플라인 회귀 (Spline Regression)
- 구간별로 다른 함수 적용

---

## 6.5 실습: 선형 vs 비선형 비교

### 데이터 생성
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 500

# 비선형 관계 데이터 생성
X = np.random.uniform(0, 10, n)
y_true = 2 + 3*X - 0.5*X**2 + 0.02*X**3
y = y_true + np.random.normal(0, 5, n)

df = pd.DataFrame({'X': X, 'y': y})

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='데이터')
plt.plot(sorted(X), sorted(y_true), 'r-', linewidth=2, label='실제 관계')
plt.xlabel('X')
plt.ylabel('y')
plt.title('비선형 관계 데이터')
plt.legend()
plt.show()
```

---

## 6.6 모델 1: 선형 모델

### 단순 선형 회귀
```python
# 선형 모델
X_linear = sm.add_constant(df['X'])
model_linear = sm.OLS(df['y'], X_linear).fit()

print("===== 선형 모델 =====")
print(model_linear.summary())

# 예측
y_pred_linear = model_linear.predict(X_linear)

# 평가
mse_linear = mean_squared_error(df['y'], y_pred_linear)
r2_linear = r2_score(df['y'], y_pred_linear)

print(f"\nMSE: {mse_linear:.2f}")
print(f"R²: {r2_linear:.4f}")
```

---

## 6.7 모델 2: 다항 회귀

### 2차 다항식
```python
# 다항 회귀 (2차)
df['X2'] = df['X'] ** 2

X_poly2 = sm.add_constant(df[['X', 'X2']])
model_poly2 = sm.OLS(df['y'], X_poly2).fit()

print("===== 2차 다항 모델 =====")
print(model_poly2.summary())

# 예측 및 평가
y_pred_poly2 = model_poly2.predict(X_poly2)
mse_poly2 = mean_squared_error(df['y'], y_pred_poly2)
r2_poly2 = r2_score(df['y'], y_pred_poly2)

print(f"\nMSE: {mse_poly2:.2f}")
print(f"R²: {r2_poly2:.4f}")
```

---

## 6.8 모델 3: 3차 다항 회귀

### 3차 다항식
```python
# 다항 회귀 (3차)
df['X3'] = df['X'] ** 3

X_poly3 = sm.add_constant(df[['X', 'X2', 'X3']])
model_poly3 = sm.OLS(df['y'], X_poly3).fit()

print("===== 3차 다항 모델 =====")
print(model_poly3.summary())

# 예측 및 평가
y_pred_poly3 = model_poly3.predict(X_poly3)
mse_poly3 = mean_squared_error(df['y'], y_pred_poly3)
r2_poly3 = r2_score(df['y'], y_pred_poly3)

print(f"\nMSE: {mse_poly3:.2f}")
print(f"R²: {r2_poly3:.4f}")
```

---

## 6.9 모델 비교 시각화

### 예측 결과 비교
```python
# 정렬된 X 값으로 예측선 그리기
X_sorted = np.sort(df['X'])
X_sorted_df = pd.DataFrame({'X': X_sorted, 'X2': X_sorted**2, 'X3': X_sorted**3})

y_pred_linear_sorted = model_linear.predict(sm.add_constant(X_sorted))
y_pred_poly2_sorted = model_poly2.predict(sm.add_constant(X_sorted_df[['X', 'X2']]))
y_pred_poly3_sorted = model_poly3.predict(sm.add_constant(X_sorted_df))

plt.figure(figsize=(12, 7))
plt.scatter(df['X'], df['y'], alpha=0.3, label='데이터')
plt.plot(X_sorted, y_pred_linear_sorted, 'g-', linewidth=2, label='선형')
plt.plot(X_sorted, y_pred_poly2_sorted, 'b-', linewidth=2, label='2차 다항')
plt.plot(X_sorted, y_pred_poly3_sorted, 'r-', linewidth=2, label='3차 다항')
plt.xlabel('X')
plt.ylabel('y')
plt.title('선형 vs 비선형 모델 비교')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 6.10 모델 성능 비교표

### 종합 평가
```python
# 모델 비교표
comparison = pd.DataFrame({
    'Model': ['선형', '2차 다항', '3차 다항'],
    'MSE': [mse_linear, mse_poly2, mse_poly3],
    'R²': [r2_linear, r2_poly2, r2_poly3],
    'Adj_R²': [model_linear.rsquared_adj, 
               model_poly2.rsquared_adj, 
               model_poly3.rsquared_adj],
    'AIC': [model_linear.aic, model_poly2.aic, model_poly3.aic],
    'BIC': [model_linear.bic, model_poly2.bic, model_poly3.bic]
})

print("===== 모델 성능 비교 =====")
print(comparison)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# MSE
comparison.plot(x='Model', y='MSE', kind='bar', ax=axes[0], legend=False, color='skyblue')
axes[0].set_title('Mean Squared Error')
axes[0].set_ylabel('MSE')

# R²
comparison.plot(x='Model', y='R²', kind='bar', ax=axes[1], legend=False, color='lightgreen')
axes[1].set_title('R-squared')
axes[1].set_ylabel('R²')

# AIC
comparison.plot(x='Model', y='AIC', kind='bar', ax=axes[2], legend=False, color='coral')
axes[2].set_title('AIC (낮을수록 좋음)')
axes[2].set_ylabel('AIC')

plt.tight_layout()
plt.show()
```

---

## 6.11 정보 기준 (Information Criteria)

### AIC vs BIC

#### AIC (Akaike Information Criterion)
```
AIC = -2log(L) + 2k
```
- L: 최대 우도 (likelihood)
- k: 모수 개수
- **낮을수록 좋음**

#### BIC (Bayesian Information Criterion)
```
BIC = -2log(L) + k×log(n)
```
- n: 표본 크기
- **BIC가 AIC보다 복잡성에 더 큰 패널티**

### 선택 기준
- AIC: 예측에 중점
- BIC: 더 간결한 모델 선호

---

## 6.12 교차 검증 (Cross-Validation)

### K-Fold Cross-Validation
```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 데이터 준비
X_array = df['X'].values.reshape(-1, 1)
y_array = df['y'].values

# K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 각 차수별로 CV 수행
degrees = [1, 2, 3, 4, 5]
cv_scores = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_array)
    
    fold_scores = []
    for train_idx, test_idx in kf.split(X_poly):
        X_train, X_test = X_poly[train_idx], X_poly[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        fold_scores.append(score)
    
    cv_scores.append(np.mean(fold_scores))
    print(f"차수 {degree}: CV R² = {np.mean(fold_scores):.4f}")

# 최적 차수
best_degree = degrees[np.argmax(cv_scores)]
print(f"\n최적 차수: {best_degree}")
```

---

## 6.13 교차 검증 결과 시각화

### CV 점수 비교
```python
plt.figure(figsize=(10, 6))
plt.plot(degrees, cv_scores, 'o-', linewidth=2, markersize=10)
plt.xlabel('다항식 차수')
plt.ylabel('평균 CV R²')
plt.title('교차 검증: 모델 복잡도 vs 성능')
plt.grid(True, alpha=0.3)
plt.axvline(x=best_degree, color='r', linestyle='--', 
            label=f'최적 차수: {best_degree}')
plt.legend()
plt.show()
```

---

## 6.14 패널 vs 횡단면 데이터

### 데이터 구조 비교

#### 횡단면 데이터 (Cross-Sectional)
- 한 시점의 여러 개체
- **모델**: OLS

#### 패널 데이터 (Panel)
- 여러 개체를 시간에 걸쳐 관찰
- **모델**: Pooled OLS, FE, RE

### 선택 기준
- 시간 불변 이질성 통제 필요? → 패널 (FE/RE)
- 동태적 관계 분석? → 패널
- 한 시점만 있음? → 횡단면

---

## 6.15 실습: 패널 vs 횡단면 비교

### 데이터 생성
```python
np.random.seed(42)
n_firms = 100
n_years = 5

# 패널 데이터 생성
firms = np.repeat(range(1, n_firms+1), n_years)
years = np.tile(range(2019, 2024), n_firms)

# 기업 고정효과 (관측 불가)
firm_effects = np.random.normal(50, 15, n_firms)
firm_effects_exp = np.repeat(firm_effects, n_years)

# 시간 효과
time_effects = np.tile([0, 2, 4, 3, 5], n_firms)

df_panel = pd.DataFrame({
    'firm_id': firms,
    'year': years,
    'investment': np.random.uniform(10, 50, n_firms*n_years),
    'market_share': np.random.uniform(0, 20, n_firms*n_years)
})

# 종속변수 (매출)
df_panel['sales'] = (firm_effects_exp + 
                     time_effects +
                     2.5 * df_panel['investment'] + 
                     3.0 * df_panel['market_share'] + 
                     np.random.normal(0, 10, n_firms*n_years))

# 횡단면 데이터 (2023년만)
df_cross = df_panel[df_panel['year'] == 2023].copy()

print(f"패널 데이터 크기: {df_panel.shape}")
print(f"횡단면 데이터 크기: {df_cross.shape}")
```

---

## 6.16 횡단면 OLS

### 2023년 데이터만 사용
```python
# 횡단면 OLS
X_cross = sm.add_constant(df_cross[['investment', 'market_share']])
y_cross = df_cross['sales']

model_cross = sm.OLS(y_cross, X_cross).fit()
print("===== 횡단면 OLS (2023년만) =====")
print(model_cross.summary())
```

### 문제점
- 기업 고정효과 무시
- 편향된 추정
- 시간에 따른 변화 포착 못함

---

## 6.17 패널 Pooled OLS

### 모든 연도 합쳐서 분석
```python
# Pooled OLS
X_pooled = sm.add_constant(df_panel[['investment', 'market_share']])
y_pooled = df_panel['sales']

model_pooled = sm.OLS(y_pooled, X_pooled).fit()
print("===== Pooled OLS (모든 연도) =====")
print(model_pooled.summary())
```

### 문제점
- 여전히 기업 고정효과 무시
- 관측치의 독립성 가정 위배 (같은 기업의 여러 관측치)

---

## 6.18 패널 Fixed Effects

### 기업 고정효과 통제
```python
from linearmodels.panel import PanelOLS

# 데이터 인덱싱
df_panel_indexed = df_panel.set_index(['firm_id', 'year'])

# Fixed Effects
fe_model = PanelOLS.from_formula(
    'sales ~ investment + market_share + EntityEffects',
    data=df_panel_indexed
)
fe_results = fe_model.fit()

print("===== Fixed Effects =====")
print(fe_results)
```

### 장점
- 기업별 이질성 통제
- 일치 추정량 (consistent)

---

## 6.19 모델 비교: 횡단면 vs 패널

### 계수 비교
```python
print("===== 모델별 계수 비교 =====")
print(f"{'Model':<15} {'Investment':<12} {'Market Share':<12}")
print("-" * 40)
print(f"{'Cross-section':<15} {model_cross.params['investment']:<12.4f} {model_cross.params['market_share']:<12.4f}")
print(f"{'Pooled OLS':<15} {model_pooled.params['investment']:<12.4f} {model_pooled.params['market_share']:<12.4f}")
print(f"{'Fixed Effects':<15} {fe_results.params['investment']:<12.4f} {fe_results.params['market_share']:<12.4f}")

# 시각화
models_names = ['횡단면\n(2023)', 'Pooled\nOLS', 'Fixed\nEffects']
inv_coefs = [model_cross.params['investment'], 
             model_pooled.params['investment'], 
             fe_results.params['investment']]
mkt_coefs = [model_cross.params['market_share'], 
             model_pooled.params['market_share'], 
             fe_results.params['market_share']]

x = np.arange(len(models_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, inv_coefs, width, label='Investment', alpha=0.8)
ax.bar(x + width/2, mkt_coefs, width, label='Market Share', alpha=0.8)

ax.set_ylabel('계수 추정값')
ax.set_title('횡단면 vs 패널 모델 계수 비교')
ax.set_xticks(x)
ax.set_xticklabels(models_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 6.20 강건성 테스트 (Robustness Checks)

### 강건성 테스트란?
- 모델의 결과가 다양한 설정에서도 일관되는지 확인
- 연구 결과의 신뢰성 제고

### 주요 방법

#### 1. 표본 변경
- 이상치 제거
- 기간 변경
- 특정 그룹 제외

#### 2. 변수 변경
- 종속변수 측정 방법 변경
- 통제변수 추가/제거
- 변수 변환 (로그, 표준화 등)

#### 3. 모델 변경
- 다른 추정 방법
- 클러스터 표준오차
- 강건 표준오차

---

## 6.21 강건성 테스트 예시 1: 표본 변경

### 이상치 제거 후 재분석
```python
# 원본 모델 (Chapter 5 임금 데이터 사용)
np.random.seed(42)
n = 2000

df_wage = pd.DataFrame({
    '교육년수': np.random.choice(range(9, 21), n),
    '경력년수': np.random.uniform(0, 40, n),
    '성별_더미': np.random.choice([0, 1], n)
})

df_wage['log_임금'] = (1.5 + 
                       0.08 * df_wage['교육년수'] + 
                       0.04 * df_wage['경력년수'] +
                       0.15 * df_wage['성별_더미'] +
                       np.random.normal(0, 0.3, n))

# 기본 모델
X_base = sm.add_constant(df_wage[['교육년수', '경력년수', '성별_더미']])
y_base = df_wage['log_임금']
model_base = sm.OLS(y_base, X_base).fit()

print("===== 기본 모델 =====")
print(f"교육년수 계수: {model_base.params['교육년수']:.4f} (p={model_base.pvalues['교육년수']:.4f})")

# 이상치 제거 (log_임금의 상하위 5%)
Q1 = df_wage['log_임금'].quantile(0.05)
Q3 = df_wage['log_임금'].quantile(0.95)
df_robust = df_wage[(df_wage['log_임금'] >= Q1) & (df_wage['log_임금'] <= Q3)]

# 강건성 모델
X_robust = sm.add_constant(df_robust[['교육년수', '경력년수', '성별_더미']])
y_robust = df_robust['log_임금']
model_robust = sm.OLS(y_robust, X_robust).fit()

print("\n===== 이상치 제거 후 =====")
print(f"교육년수 계수: {model_robust.params['교육년수']:.4f} (p={model_robust.pvalues['교육년수']:.4f})")
print(f"\n표본 크기: {n} → {len(df_robust)}")
```

---

## 6.22 강건성 테스트 예시 2: 변수 변경

### 통제변수 추가
```python
# 추가 변수 생성
df_wage['산업'] = np.random.choice(['제조업', '서비스업', 'IT'], n)
df_wage['산업_제조업'] = (df_wage['산업'] == '제조업').astype(int)
df_wage['산업_IT'] = (df_wage['산업'] == 'IT').astype(int)

# 통제변수 추가 모델
X_controls = sm.add_constant(df_wage[['교육년수', '경력년수', '성별_더미', 
                                      '산업_제조업', '산업_IT']])
y_controls = df_wage['log_임금']
model_controls = sm.OLS(y_controls, X_controls).fit()

print("===== 통제변수 추가 모델 =====")
print(f"교육년수 계수: {model_controls.params['교육년수']:.4f} (p={model_controls.pvalues['교육년수']:.4f})")

# 비교
print("\n===== 강건성 체크: 교육년수 계수 =====")
print(f"기본 모델:        {model_base.params['교육년수']:.4f}")
print(f"이상치 제거:      {model_robust.params['교육년수']:.4f}")
print(f"통제변수 추가:    {model_controls.params['교육년수']:.4f}")
print("\n→ 모든 모델에서 일관된 양(+)의 효과 확인")
```

---

## 6.23 강건성 테스트 예시 3: 강건 표준오차

### Heteroskedasticity-Robust Standard Errors
```python
# 일반 표준오차
model_ols = sm.OLS(y_base, X_base).fit()

# 강건 표준오차 (HC3)
model_robust_se = sm.OLS(y_base, X_base).fit(cov_type='HC3')

print("===== 표준오차 비교 =====")
print("\n일반 OLS:")
print(model_ols.summary().tables[1])

print("\n강건 표준오차 (HC3):")
print(model_robust_se.summary().tables[1])
```

---

## 6.24 클러스터 표준오차

### Clustered Standard Errors
```python
# 패널 데이터에서 기업별 클러스터링
from linearmodels.panel import PanelOLS

# 클러스터 표준오차
fe_cluster = PanelOLS.from_formula(
    'sales ~ investment + market_share + EntityEffects',
    data=df_panel_indexed
).fit(cov_type='clustered', cluster_entity=True)

print("===== 클러스터 표준오차 =====")
print(fe_cluster)
```

### 언제 사용?
- 같은 클러스터 내 관측치의 상관관계
- 예: 같은 기업, 같은 지역, 같은 시점

---

## 6.25 예측력 평가 지표

### 회귀 모델 평가 지표

#### 1. MSE (Mean Squared Error)
```python
MSE = Σ(yᵢ - ŷᵢ)² / n
```
- 단위: 종속변수 제곱
- 이상치에 민감

#### 2. RMSE (Root Mean Squared Error)
```python
RMSE = √MSE
```
- 단위: 종속변수와 동일
- 해석 용이

#### 3. MAE (Mean Absolute Error)
```python
MAE = Σ|yᵢ - ŷᵢ| / n
```
- 이상치에 덜 민감
- 절대 오차의 평균

---

## 6.26 예측력 평가 지표 (계속)

### 추가 지표

#### 4. MAPE (Mean Absolute Percentage Error)
```python
MAPE = (100/n) × Σ|yᵢ - ŷᵢ| / |yᵢ|
```
- 백분율로 표현
- 스케일 독립적

#### 5. R² (Coefficient of Determination)
```python
R² = 1 - (SS_res / SS_tot)
```
- 0~1 사이 값
- 설명력 나타냄

---

## 6.27 실습: 예측력 평가

### Train-Test Split
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 데이터 분할
X = df_wage[['교육년수', '경력년수', '성별_더미']]
y = df_wage['log_임금']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 학습
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

model_pred = sm.OLS(y_train, X_train_const).fit()

# 예측
y_train_pred = model_pred.predict(X_train_const)
y_test_pred = model_pred.predict(X_test_const)
```

---

## 6.28 예측 성능 계산

### In-sample vs Out-of-sample
```python
# In-sample (학습 데이터)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Out-of-sample (테스트 데이터)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 결과 정리
results = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
    'Train': [train_mse, train_rmse, train_mae, train_r2],
    'Test': [test_mse, test_rmse, test_mae, test_r2]
})

print("===== 예측 성능 비교 =====")
print(results)

# 과적합 체크
print(f"\n과적합 체크:")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²:  {test_r2:.4f}")
print(f"차이: {train_r2 - test_r2:.4f}")

if train_r2 - test_r2 > 0.1:
    print("→ 과적합 가능성 있음")
else:
    print("→ 과적합 문제 없음")
```

---

## 6.29 예측 시각화

### 실제 vs 예측
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train set
axes[0].scatter(y_train, y_train_pred, alpha=0.5)
axes[0].plot([y_train.min(), y_train.max()], 
             [y_train.min(), y_train.max()], 
             'r--', linewidth=2)
axes[0].set_xlabel('실제 log(임금)')
axes[0].set_ylabel('예측 log(임금)')
axes[0].set_title(f'Train Set (R²={train_r2:.4f})')
axes[0].grid(True, alpha=0.3)

# Test set
axes[1].scatter(y_test, y_test_pred, alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', linewidth=2)
axes[1].set_xlabel('실제 log(임금)')
axes[1].set_ylabel('예측 log(임금)')
axes[1].set_title(f'Test Set (R²={test_r2:.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 6.30 잔차 분석 - 예측 모델

### 잔차 패턴 확인
```python
# 테스트 데이터 잔차
test_residuals = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 잔차 플롯
axes[0].scatter(y_test_pred, test_residuals, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('예측값')
axes[0].set_ylabel('잔차')
axes[0].set_title('잔차 플롯 (Test Set)')
axes[0].grid(True, alpha=0.3)

# 잔차 히스토그램
axes[1].hist(test_residuals, bins=30, edgecolor='black')
axes[1].set_xlabel('잔차')
axes[1].set_ylabel('빈도')
axes[1].set_title('잔차 분포')
axes[1].axvline(x=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()

# 잔차 통계
print("===== 잔차 통계 =====")
print(f"평균: {test_residuals.mean():.6f}")
print(f"표준편차: {test_residuals.std():.4f}")
print(f"왜도: {test_residuals.skew():.4f}")
print(f"첨도: {test_residuals.kurtosis():.4f}")
```

---

## 6.31 경제적 해석력 높이기

### 해석 개선 방법

#### 1. 표준화 계수
- 변수 간 영향력 크기 비교
- 단위에 독립적

#### 2. 탄력성
- 1% 변화에 대한 반응
- 로그-로그 모델

#### 3. 한계효과
- 1단위 변화의 실질적 영향
- 평균값에서 계산

#### 4. 시나리오 분석
- "만약 X가 Y만큼 변한다면?"
- 정책 시뮬레이션

---

## 6.32 표준화 계수 계산

### 비교 가능한 계수
```python
from sklearn.preprocessing import StandardScaler

# 변수 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 표준화된 데이터로 회귀
X_scaled_const = sm.add_constant(X_scaled_df)
model_std = sm.OLS(y, X_scaled_const).fit()

# 원본 vs 표준화 계수
print("===== 계수 비교 =====")
print("\n원본 계수:")
print(model_base.params[1:])

print("\n표준화 계수:")
print(model_std.params[1:])

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 원본 계수
model_base.params[1:].plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_title('원본 계수')
axes[0].set_xlabel('계수값')

# 표준화 계수
model_std.params[1:].plot(kind='barh', ax=axes[1], color='lightcoral')
axes[1].set_title('표준화 계수 (비교 가능)')
axes[1].set_xlabel('표준화 계수')

plt.tight_layout()
plt.show()

print("\n해석: 표준화 계수가 클수록 영향력이 크다")
```

---

## 6.33 한계효과 계산

### 평균값에서의 한계효과
```python
# 평균값
mean_edu = df_wage['교육년수'].mean()
mean_exp = df_wage['경력년수'].mean()

print("===== 한계효과 분석 =====")
print(f"\n평균 교육년수: {mean_edu:.2f}년")
print(f"평균 경력년수: {mean_exp:.2f}년")

# 교육 1년 증가의 효과
edu_effect = model_base.params['교육년수']
wage_increase_pct = (np.exp(edu_effect) - 1) * 100

print(f"\n교육년수 1년 증가 효과:")
print(f"- 로그 임금 변화: {edu_effect:.4f}")
print(f"- 실제 임금 변화: {wage_increase_pct:.2f}% 증가")

# 평균 임금이 30,000원이라면
avg_wage = 30000
wage_increase = avg_wage * (np.exp(edu_effect) - 1)
print(f"- 시간당 {avg_wage:,}원일 때: {wage_increase:,.0f}원 증가")
print(f"- 월급 환산 (주40시간): {wage_increase*40*4.33:,.0f}원 증가")
```

---

## 6.34 시나리오 분석

### 정책 시뮬레이션
```python
# 시나리오: 교육년수 증가 정책
scenarios = pd.DataFrame({
    '시나리오': ['현재', '고졸→전문대', '고졸→대졸', '전문대→대졸'],
    '교육년수_변화': [0, 2, 4, 2],
    '교육년수_전': [12, 12, 12, 14],
    '교육년수_후': [12, 14, 16, 16]
})

# 임금 변화 계산
for idx, row in scenarios.iterrows():
    edu_change = row['교육년수_변화']
    if edu_change > 0:
        log_wage_change = edu_change * model_base.params['교육년수']
        wage_change_pct = (np.exp(log_wage_change) - 1) * 100
        scenarios.loc[idx, '임금증가율(%)'] = wage_change_pct
        
        # 평균 임금 기준 금액
        avg_wage = 30000
        wage_increase = avg_wage * (np.exp(log_wage_change) - 1)
        scenarios.loc[idx, '시간당증가(원)'] = wage_increase
        scenarios.loc[idx, '월급증가(원)'] = wage_increase * 40 * 4.33
    else:
        scenarios.loc[idx, '임금증가율(%)'] = 0
        scenarios.loc[idx, '시간당증가(원)'] = 0
        scenarios.loc[idx, '월급증가(원)'] = 0

print("===== 교육 투자 시나리오 분석 =====")
print(scenarios.to_string(index=False))
```

---

## 6.35 시나리오 분석 시각화

### 교육 투자 수익
```python
plt.figure(figsize=(12, 6))

x = np.arange(len(scenarios))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 임금 증가율
axes[0].bar(x, scenarios['임금증가율(%)'], color='steelblue')
axes[0].set_xlabel('시나리오')
axes[0].set_ylabel('임금 증가율 (%)')
axes[0].set_title('교육년수 증가에 따른 임금 증가율')
axes[0].set_xticks(x)
axes[0].set_xticklabels(scenarios['시나리오'], rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)

# 값 표시
for i, v in enumerate(scenarios['임금증가율(%)']):
    if v > 0:
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

# 월급 증가액
axes[1].bar(x, scenarios['월급증가(원)']/10000, color='coral')
axes[1].set_xlabel('시나리오')
axes[1].set_ylabel('월급 증가 (만원)')
axes[1].set_title('교육년수 증가에 따른 월급 증가액')
axes[1].set_xticks(x)
axes[1].set_xticklabels(scenarios['시나리오'], rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)

# 값 표시
for i, v in enumerate(scenarios['월급증가(원)']):
    if v > 0:
        axes[1].text(i, v/10000 + 5, f'{v/10000:.0f}만원', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

---

## 6.36 민감도 분석

### 계수 변화에 따른 영향
```python
# 교육 수익률의 불확실성 고려
edu_coef_lower = model_base.conf_int().loc['교육년수', 0]  # 95% CI 하한
edu_coef_upper = model_base.conf_int().loc['교육년수', 1]  # 95% CI 상한
edu_coef_estimate = model_base.params['교육년수']

print("===== 민감도 분석 =====")
print(f"\n교육년수 계수:")
print(f"추정값: {edu_coef_estimate:.4f}")
print(f"95% 신뢰구간: [{edu_coef_lower:.4f}, {edu_coef_upper:.4f}]")

# 대졸(16년) vs 고졸(12년) 시나리오
edu_diff = 4

scenarios_sensitivity = pd.DataFrame({
    '시나리오': ['비관적 (CI 하한)', '추정값', '낙관적 (CI 상한)'],
    '계수': [edu_coef_lower, edu_coef_estimate, edu_coef_upper]
})

for idx, row in scenarios_sensitivity.iterrows():
    log_wage_diff = edu_diff * row['계수']
    wage_diff_pct = (np.exp(log_wage_diff) - 1) * 100
    scenarios_sensitivity.loc[idx, '임금차이(%)'] = wage_diff_pct

print("\n대졸 vs 고졸 임금 차이 (4년 교육 차이):")
print(scenarios_sensitivity)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(scenarios_sensitivity['시나리오'], 
         scenarios_sensitivity['임금차이(%)'],
         color=['lightcoral', 'steelblue', 'lightgreen'])
plt.xlabel('임금 차이 (%)')
plt.title('교육 수익률의 민감도 분석 (대졸 vs 고졸)')
plt.grid(axis='x', alpha=0.3)

for i, v in enumerate(scenarios_sensitivity['임금차이(%)']):
    plt.text(v + 1, i, f'{v:.1f}%', va='center')

plt.tight_layout()
plt.show()
```

---

## 6.37 모델 진단 체크리스트

### 분석 전 체크리스트

- [ ] 데이터 품질 확인 (결측치, 이상치)
- [ ] 변수 간 상관관계 확인
- [ ] 종속변수 분포 확인

### 모델 추정 후 체크리스트

- [ ] 잔차의 정규성 (Q-Q plot)
- [ ] 등분산성 (residual plot)
- [ ] 다중공선성 (VIF < 10)
- [ ] 영향력 관측치 (Cook's distance)

### 결과 보고 전 체크리스트

- [ ] 강건성 테스트 수행
- [ ] 예측력 평가 (out-of-sample)
- [ ] 경제적 해석 추가
- [ ] 한계점 명시

---

## 6.38 Best Practices

### 모델 선택 가이드

#### 1. 간결성의 원칙 (Parsimony)
- 불필요한 변수 제거
- Occam's Razor: 단순한 모델 선호

#### 2. 이론 기반 선택
- 경제학 이론에 근거
- 선행 연구 참고

#### 3. 데이터 기반 검증
- 교차 검증
- 정보 기준 (AIC, BIC)

#### 4. 강건성 확인
- 여러 설정에서 일관된 결과
- 민감도 분석

---

## 6.39 흔한 실수들

### 피해야 할 실수

#### 1. 과적합 (Overfitting)
- 너무 많은 변수 포함
- 해결: 교차 검증, 정규화

#### 2. p-hacking
- 유의한 결과 나올 때까지 변수 조합
- 해결: 사전 분석 계획, 강건성 테스트

#### 3. 내생성 무시
- X와 오차항의 상관관계
- 해결: 도구변수, 고정효과

#### 4. 표준오차 오류
- 이분산성, 자기상관 무시
- 해결: 강건 표준오차, 클러스터링

---

## 6.40 Chapter 6 요약

### 배운 내용
✓ 선형 vs 비선형 모델 비교  
✓ 다항 회귀와 교차 검증  
✓ 패널 vs 횡단면 데이터 분석  
✓ 강건성 테스트 방법  
✓ 예측력 평가 지표 (MSE, RMSE, MAE, R²)  
✓ 경제적 해석력 (표준화 계수, 한계효과, 시나리오 분석)  
✓ 민감도 분석  
✓ 모델 진단 체크리스트

### 핵심 메시지
- 최적 모델은 맥락에 따라 다름
- 강건성 테스트는 필수
- 예측력과 해석력의 균형

---

## 6.41 실습 과제

### 과제 1: 모델 비교
- 선형, 2차, 3차 다항 모델 비교
- AIC, BIC, CV로 최적 모델 선택
- 결과 해석 및 시각화

### 과제 2: 강건성 테스트
- 표본 변경 (기간, 이상치 제거)
- 변수 변경 (통제변수 추가/제거)
- 모델 변경 (강건 표준오차)
- 결과의 일관성 확인

### 과제 3: 예측 및 시나리오 분석
- Train-test split으로 예측력 평가
- 정책 시나리오 3가지 설계
- 한계효과 및 민감도 분석
- 종합 보고서 작성

---

## 6.42 다음 단계

### Chapter 7 예고: 머신러닝을 활용한 계량경제학적 확장
- 계량경제학 vs 머신러닝
- Random Forest, Gradient Boosting
- 변수 중요도 분석
- SHAP values
- 계량경제학과 ML의 융합

### 준비사항
- Scikit-learn 라이브러리 설치
- 머신러닝 기초 개념 예습
- 과적합과 정규화 이해

---

## 참고 자료

### 교재
- "The Elements of Statistical Learning" by Hastie et al.
- "Mostly Harmless Econometrics" by Angrist & Pischke

### Python 라이브러리
- `statsmodels`: 통계 모델링
- `linearmodels`: 패널 데이터
- `sklearn`: 교차 검증, 평가 지표

### 논문 작성 가이드
- AER Submission Guidelines
- Journal of Econometrics Author Guide

---

## Q&A

질문이 있으시면 편하게 물어보세요!

모델 선택에 어려움이 있나요?
강건성 테스트 방법이 궁금한가요?
