# Chapter 7: 머신러닝을 활용한 계량경제학적 확장

---

## 7.1 계량경제학 vs 머신러닝

### 접근 방식의 차이

#### 계량경제학 (Econometrics)
- **목표**: 인과관계 추론, 이론 검증
- **강점**: 해석 가능성, 통계적 추론
- **방법**: OLS, IV, 패널 모델
- **중점**: "왜?" (Why)

#### 머신러닝 (Machine Learning)
- **목표**: 예측 정확도 최대화
- **강점**: 비선형 관계 포착, 높은 예측력
- **방법**: Random Forest, Neural Networks
- **중점**: "무엇?" (What)

---

## 7.2 두 접근법의 비교

### 주요 차이점

| 측면 | 계량경제학 | 머신러닝 |
|------|------------|----------|
| **목적** | 인과추론 | 예측 |
| **모델** | 단순, 선형 | 복잡, 비선형 |
| **해석** | 용이 | 어려움 |
| **과적합** | 덜 민감 | 민감 |
| **변수선택** | 이론 기반 | 데이터 기반 |
| **표준오차** | 중요 | 덜 중요 |
| **검증** | p-value | CV, Test error |

---

## 7.3 융합의 필요성

### 왜 융합인가?

#### 계량경제학의 한계
- 선형 관계 가정
- 복잡한 상호작용 포착 어려움
- 예측력 제한

#### 머신러닝의 한계
- 인과관계 파악 어려움
- 블랙박스 문제
- 경제학 이론 무시 가능

### 융합의 장점
✓ 높은 예측력 + 경제적 해석  
✓ 변수 중요도 파악  
✓ 복잡한 비선형 관계 발견  
✓ 정책 시뮬레이션 정확도 향상

---

## 7.4 머신러닝 기초 개념

### 주요 개념

#### 1. 지도 학습 (Supervised Learning)
- 입력(X)과 출력(y) 데이터로 학습
- **회귀**: 연속형 예측 (가격, 소득)
- **분류**: 범주형 예측 (승인/거절)

#### 2. 비지도 학습 (Unsupervised Learning)
- 출력 없이 패턴 발견
- 클러스터링, 차원 축소

#### 3. 과적합 (Overfitting)
- 학습 데이터에 너무 맞춤
- 새 데이터 예측력 저하

#### 4. 정규화 (Regularization)
- 과적합 방지
- Ridge, Lasso, Elastic Net

---

## 7.5 Scikit-learn 소개

### 머신러닝의 표준 라이브러리

```python
# 설치
pip install scikit-learn

# 기본 임포트
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

### 주요 모듈
- `model_selection`: 데이터 분할, 교차 검증
- `preprocessing`: 데이터 전처리
- `linear_model`: 선형 모델
- `ensemble`: 앙상블 모델
- `metrics`: 평가 지표

---

## 7.6 실습 데이터 준비

### 임금 예측 데이터
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 2000

# 특성 생성
df = pd.DataFrame({
    '교육년수': np.random.choice(range(9, 21), n),
    '경력년수': np.random.uniform(0, 40, n),
    '나이': np.random.uniform(22, 65, n),
    '성별': np.random.choice([0, 1], n),
    '도시거주': np.random.choice([0, 1], n),
    '산업코드': np.random.choice(range(1, 11), n),
    '직급': np.random.choice(range(1, 6), n),
    '회사규모': np.random.uniform(10, 10000, n)
})

# 비선형 관계로 임금 생성
df['log_임금'] = (
    1.5 + 
    0.08 * df['교육년수'] +
    0.04 * df['경력년수'] - 0.0005 * df['경력년수']**2 +
    0.02 * df['나이'] +
    0.15 * df['성별'] +
    0.10 * df['도시거주'] +
    0.03 * df['산업코드'] +
    0.08 * df['직급'] +
    0.00002 * df['회사규모'] +
    0.01 * df['교육년수'] * df['성별'] +  # 상호작용
    np.random.normal(0, 0.3, n)
)

df['임금'] = np.exp(df['log_임금']) * 10000

print(df.head())
print(f"\n데이터 크기: {df.shape}")
print(f"평균 임금: {df['임금'].mean():,.0f}원")
```

---

## 7.7 데이터 분할

### Train-Test Split
```python
from sklearn.model_selection import train_test_split

# 특성과 타겟 분리
features = ['교육년수', '경력년수', '나이', '성별', '도시거주', 
            '산업코드', '직급', '회사규모']
X = df[features]
y = df['log_임금']

# 80:20 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train 크기: {X_train.shape}")
print(f"Test 크기: {X_test.shape}")
```

### 왜 분할하나?
- 과적합 방지
- 실제 성능 평가
- 일반화 능력 확인

---

## 7.8 Baseline: 선형 회귀

### Scikit-learn으로 선형 회귀
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 모델 학습
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 예측
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

# 평가
train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)
test_mse_lr = mean_squared_error(y_test, y_test_pred_lr)
train_r2_lr = r2_score(y_train, y_train_pred_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)

print("===== 선형 회귀 =====")
print(f"Train MSE: {train_mse_lr:.4f}")
print(f"Test MSE:  {test_mse_lr:.4f}")
print(f"Train R²:  {train_r2_lr:.4f}")
print(f"Test R²:   {test_r2_lr:.4f}")

# 계수 확인
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_
})
print("\n계수:")
print(coef_df.sort_values('Coefficient', key=abs, ascending=False))
```

---

## 7.9 Ridge 회귀 (L2 정규화)

### 정규화로 과적합 방지
```python
from sklearn.linear_model import Ridge

# Ridge 회귀
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# 예측 및 평가
y_test_pred_ridge = ridge_model.predict(X_test)
test_r2_ridge = r2_score(y_test, y_test_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)

print("===== Ridge 회귀 =====")
print(f"Test MSE: {test_mse_ridge:.4f}")
print(f"Test R²:  {test_r2_ridge:.4f}")
```

### Ridge vs OLS
- Ridge: 계수를 0에 가깝게 축소
- 다중공선성 완화
- alpha: 정규화 강도 (하이퍼파라미터)

---

## 7.10 Lasso 회귀 (L1 정규화)

### 변수 선택 기능
```python
from sklearn.linear_model import Lasso

# Lasso 회귀
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)

# 예측 및 평가
y_test_pred_lasso = lasso_model.predict(X_test)
test_r2_lasso = r2_score(y_test, y_test_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)

print("===== Lasso 회귀 =====")
print(f"Test MSE: {test_mse_lasso:.4f}")
print(f"Test R²:  {test_r2_lasso:.4f}")

# 선택된 변수 (계수가 0이 아닌 것)
coef_lasso = pd.DataFrame({
    'Feature': features,
    'Coefficient': lasso_model.coef_
})
selected = coef_lasso[coef_lasso['Coefficient'] != 0]
print(f"\n선택된 변수 수: {len(selected)} / {len(features)}")
print(selected.sort_values('Coefficient', key=abs, ascending=False))
```

### Lasso의 특징
- 일부 계수를 정확히 0으로 만듦
- 자동 변수 선택
- 해석 가능한 모델

---

## 7.11 Decision Tree (의사결정나무)

### 비선형 관계 학습
```python
from sklearn.tree import DecisionTreeRegressor

# Decision Tree
tree_model = DecisionTreeRegressor(max_depth=10, random_state=42)
tree_model.fit(X_train, y_train)

# 예측 및 평가
y_train_pred_tree = tree_model.predict(X_train)
y_test_pred_tree = tree_model.predict(X_test)

train_r2_tree = r2_score(y_train, y_train_pred_tree)
test_r2_tree = r2_score(y_test, y_test_pred_tree)

print("===== Decision Tree =====")
print(f"Train R²: {train_r2_tree:.4f}")
print(f"Test R²:  {test_r2_tree:.4f}")
print(f"과적합 정도: {train_r2_tree - test_r2_tree:.4f}")

# 변수 중요도
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': tree_model.feature_importances_
})
print("\n변수 중요도:")
print(importance_df.sort_values('Importance', ascending=False))
```

---

## 7.12 Random Forest (랜덤포레스트)

### 앙상블 학습의 강자
```python
from sklearn.ensemble import RandomForestRegressor

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,      # 트리 개수
    max_depth=15,          # 최대 깊이
    min_samples_split=10,  # 분할 최소 샘플
    random_state=42,
    n_jobs=-1              # 병렬 처리
)

rf_model.fit(X_train, y_train)

# 예측 및 평가
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)
test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)
test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)

print("===== Random Forest =====")
print(f"Train R²: {train_r2_rf:.4f}")
print(f"Test R²:  {test_r2_rf:.4f}")
print(f"Test MSE: {test_mse_rf:.4f}")
print(f"Test MAE: {test_mae_rf:.4f}")
```

---

## 7.13 Random Forest의 장점

### 왜 Random Forest인가?

#### 장점
✓ 비선형 관계 자동 학습  
✓ 변수 간 상호작용 포착  
✓ 과적합 방지 (앙상블)  
✓ 결측치에 강건  
✓ 변수 중요도 제공  
✓ 하이퍼파라미터 튜닝 덜 민감

#### 단점
✗ 계산 비용 높음  
✗ 해석 어려움  
✗ 외삽(extrapolation) 약함

### 경제 데이터 적용
- 소득 예측
- 주택가격 예측
- 신용 위험 평가
- 수요 예측

---

## 7.14 변수 중요도 분석

### Feature Importance
```python
# Random Forest 변수 중요도
importance_rf = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("===== Random Forest 변수 중요도 =====")
print(importance_rf)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(importance_rf['Feature'], importance_rf['Importance'])
plt.xlabel('중요도')
plt.title('Random Forest 변수 중요도')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### 해석
- 중요도 = 모델 예측에 기여하는 정도
- 높을수록 예측에 중요
- 경제적 정책 우선순위 결정에 활용

---

## 7.15 Gradient Boosting

### 순차적 앙상블 학습
```python
from sklearn.ensemble import GradientBoostingRegressor

# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train, y_train)

# 예측 및 평가
y_train_pred_gb = gb_model.predict(X_train)
y_test_pred_gb = gb_model.predict(X_test)

train_r2_gb = r2_score(y_train, y_train_pred_gb)
test_r2_gb = r2_score(y_test, y_test_pred_gb)
test_mse_gb = mean_squared_error(y_test, y_test_pred_gb)

print("===== Gradient Boosting =====")
print(f"Train R²: {train_r2_gb:.4f}")
print(f"Test R²:  {test_r2_gb:.4f}")
print(f"Test MSE: {test_mse_gb:.4f}")
```

### Random Forest vs Gradient Boosting
- **RF**: 병렬 학습, 빠름, 덜 민감
- **GB**: 순차 학습, 느림, 더 정확할 수 있음

---

## 7.16 XGBoost

### 최신 Gradient Boosting
```python
# XGBoost 설치: pip install xgboost
import xgboost as xgb

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# 예측 및 평가
y_test_pred_xgb = xgb_model.predict(X_test)
test_r2_xgb = r2_score(y_test, y_test_pred_xgb)
test_mse_xgb = mean_squared_error(y_test, y_test_pred_xgb)

print("===== XGBoost =====")
print(f"Test R²:  {test_r2_xgb:.4f}")
print(f"Test MSE: {test_mse_xgb:.4f}")
```

### XGBoost의 장점
- 매우 빠른 속도
- 높은 예측 정확도
- Kaggle 경진대회 인기
- 정규화 내장

---

## 7.17 모든 모델 비교

### 성능 비교표
```python
# 모델 비교
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 
              'Decision Tree', 'Random Forest', 
              'Gradient Boosting', 'XGBoost'],
    'Test R²': [test_r2_lr, test_r2_ridge, test_r2_lasso,
                test_r2_tree, test_r2_rf, test_r2_gb, test_r2_xgb],
    'Test MSE': [test_mse_lr, test_mse_ridge, test_mse_lasso,
                 mean_squared_error(y_test, y_test_pred_tree),
                 test_mse_rf, test_mse_gb, test_mse_xgb]
})

results = results.sort_values('Test R²', ascending=False)
print("===== 모델 성능 비교 =====")
print(results.to_string(index=False))

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² 비교
axes[0].barh(results['Model'], results['Test R²'])
axes[0].set_xlabel('Test R²')
axes[0].set_title('모델별 R² 비교 (높을수록 좋음)')
axes[0].invert_yaxis()

# MSE 비교
axes[1].barh(results['Model'], results['Test MSE'])
axes[1].set_xlabel('Test MSE')
axes[1].set_title('모델별 MSE 비교 (낮을수록 좋음)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
```

---

## 7.18 예측 시각화

### 실제 vs 예측
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

models = [
    ('Linear Reg', y_test_pred_lr),
    ('Ridge', y_test_pred_ridge),
    ('Lasso', y_test_pred_lasso),
    ('Decision Tree', y_test_pred_tree),
    ('Random Forest', y_test_pred_rf),
    ('XGBoost', y_test_pred_xgb)
]

for idx, (name, y_pred) in enumerate(models):
    row = idx // 3
    col = idx % 3
    
    axes[row, col].scatter(y_test, y_pred, alpha=0.5)
    axes[row, col].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', linewidth=2)
    axes[row, col].set_xlabel('실제 log(임금)')
    axes[row, col].set_ylabel('예측 log(임금)')
    axes[row, col].set_title(f'{name} (R²={r2_score(y_test, y_pred):.4f})')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 7.19 하이퍼파라미터 튜닝

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

# Random Forest 튜닝
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10, 20]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,                    # 5-fold CV
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

print("===== Grid Search 결과 =====")
print(f"최적 파라미터: {rf_grid.best_params_}")
print(f"최적 CV R²: {rf_grid.best_score_:.4f}")

# 최적 모델로 테스트
y_test_pred_best = rf_grid.best_estimator_.predict(X_test)
test_r2_best = r2_score(y_test, y_test_pred_best)
print(f"Test R²: {test_r2_best:.4f}")
```

---

## 7.20 Cross-Validation 상세

### K-Fold CV로 안정성 평가
```python
from sklearn.model_selection import cross_val_score

# Random Forest의 안정성 평가
cv_scores = cross_val_score(
    rf_model, X_train, y_train,
    cv=10,              # 10-fold CV
    scoring='r2',
    n_jobs=-1
)

print("===== 10-Fold Cross-Validation =====")
print(f"CV R² 점수: {cv_scores}")
print(f"평균: {cv_scores.mean():.4f}")
print(f"표준편차: {cv_scores.std():.4f}")
print(f"95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, "
      f"{cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

# 시각화
plt.figure(figsize=(10, 6))
plt.boxplot([cv_scores], labels=['Random Forest'])
plt.ylabel('R² Score')
plt.title('10-Fold Cross-Validation 결과')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

---

## 7.21 Partial Dependence Plot (PDP)

### 변수의 한계효과 시각화
```python
from sklearn.inspection import PartialDependenceDisplay

# PDP 생성
features_to_plot = [0, 1, 2]  # 교육년수, 경력년수, 나이
fig, ax = plt.subplots(figsize=(14, 4))

PartialDependenceDisplay.from_estimator(
    rf_model, X_train, features_to_plot,
    feature_names=features,
    ax=ax,
    n_jobs=-1
)
plt.tight_layout()
plt.show()

# 해석
print("===== PDP 해석 =====")
print("- 각 변수가 예측에 미치는 평균적 영향")
print("- 다른 변수들을 평균값으로 고정")
print("- 비선형 관계를 시각적으로 확인")
```

---

## 7.22 SHAP Values

### 모델 해석의 최신 기법
```python
# SHAP 설치: pip install shap
import shap

# SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=features, 
                  show=False)
plt.tight_layout()
plt.show()

# 특정 샘플 설명
idx = 0  # 첫 번째 테스트 샘플
plt.figure()
shap.waterfall_plot(shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[idx],
    feature_names=features
))
plt.tight_layout()
plt.show()
```

---

## 7.23 SHAP 해석

### SHAP의 장점

#### 게임 이론 기반
- 각 변수의 기여도를 공정하게 할당
- Shapley value 활용

#### 해석
- **빨간색**: 높은 특성값 → 예측 증가
- **파란색**: 낮은 특성값 → 예측 감소
- **위치**: 변수 중요도 순서

### 경제학 응용
- 임금 격차 요인 분해
- 정책 효과 분석
- 투명한 의사결정

---

## 7.24 계량경제학 + ML 융합 방법

### 1. 이중 ML (Double Machine Learning)

#### 개념
- 인과효과 추정에 ML 활용
- 편향 제거 + 예측력 향상

#### 과정
```
1. ML로 Y를 다른 변수로 예측 → 잔차
2. ML로 관심 변수 X를 다른 변수로 예측 → 잔차
3. 두 잔차로 회귀 → 인과효과
```

#### 장점
- 복잡한 통제변수 관계 처리
- 누락변수 편향 완화
- 유연한 함수형태

---

## 7.25 실습: Double ML

### Python 구현
```python
# 1단계: Y 예측 (통제변수로)
controls = ['나이', '도시거주', '산업코드', '직급', '회사규모']
X_controls = X_train[controls]
X_controls_test = X_test[controls]

rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
rf_y.fit(X_controls, y_train)

y_train_resid = y_train - rf_y.predict(X_controls)
y_test_resid = y_test - rf_y.predict(X_controls_test)

# 2단계: 교육년수 예측 (통제변수로)
edu_train = X_train['교육년수']
edu_test = X_test['교육년수']

rf_x = RandomForestRegressor(n_estimators=100, random_state=42)
rf_x.fit(X_controls, edu_train)

edu_train_resid = edu_train - rf_x.predict(X_controls)
edu_test_resid = edu_test - rf_x.predict(X_controls_test)

# 3단계: 잔차 간 회귀
from sklearn.linear_model import LinearRegression
dml_model = LinearRegression()
dml_model.fit(edu_train_resid.values.reshape(-1, 1), y_train_resid)

edu_effect = dml_model.coef_[0]
print("===== Double ML 결과 =====")
print(f"교육의 인과효과: {edu_effect:.4f}")
print(f"해석: 교육 1년 증가 → log(임금) {edu_effect:.4f} 증가")
print(f"임금 증가율: {(np.exp(edu_effect)-1)*100:.2f}%")
```

---

## 7.26 융합 방법 2: ML for Variable Selection

### ML로 중요 변수 선택 후 OLS

```python
# 1단계: Random Forest로 중요 변수 선택
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selector.fit(X_train, y_train)

# 중요도 상위 변수 선택
importance = pd.DataFrame({
    'feature': features,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

top_features = importance.head(5)['feature'].tolist()
print(f"선택된 변수: {top_features}")

# 2단계: 선택된 변수로 OLS
import statsmodels.api as sm

X_train_selected = sm.add_constant(X_train[top_features])
X_test_selected = sm.add_constant(X_test[top_features])

ols_after_ml = sm.OLS(y_train, X_train_selected).fit()
print("\n===== ML 변수선택 후 OLS =====")
print(ols_after_ml.summary())

# 예측
y_test_pred_hybrid = ols_after_ml.predict(X_test_selected)
test_r2_hybrid = r2_score(y_test, y_test_pred_hybrid)
print(f"\nTest R²: {test_r2_hybrid:.4f}")
```

---

## 7.27 융합 방법 3: Ensemble Predictions

### ML과 계량경제 모델 앙상블
```python
# 여러 모델의 예측 결합
predictions = pd.DataFrame({
    'OLS': y_test_pred_lr,
    'Ridge': y_test_pred_ridge,
    'Random Forest': y_test_pred_rf,
    'XGBoost': y_test_pred_xgb
})

# 단순 평균 앙상블
y_test_pred_ensemble_avg = predictions.mean(axis=1)
r2_ensemble_avg = r2_score(y_test, y_test_pred_ensemble_avg)

print("===== 앙상블 예측 =====")
print(f"평균 앙상블 R²: {r2_ensemble_avg:.4f}")

# 가중 평균 (성능에 비례)
weights = np.array([test_r2_lr, test_r2_ridge, test_r2_rf, test_r2_xgb])
weights = weights / weights.sum()

y_test_pred_ensemble_weighted = (predictions * weights).sum(axis=1)
r2_ensemble_weighted = r2_score(y_test, y_test_pred_ensemble_weighted)

print(f"가중 앙상블 R²: {r2_ensemble_weighted:.4f}")
print(f"\n가중치: {dict(zip(predictions.columns, weights))}")
```

---

## 7.28 실전 예시: 경제지표 예측

### GDP 성장률 예측
```python
# 시계열 특성 생성 (예시)
np.random.seed(42)
n_quarters = 100

economic_data = pd.DataFrame({
    '분기': range(1, n_quarters+1),
    '실업률': np.random.uniform(3, 8, n_quarters),
    '인플레이션': np.random.uniform(0, 5, n_quarters),
    '금리': np.random.uniform(0, 5, n_quarters),
    '환율': np.random.uniform(1000, 1400, n_quarters),
    '주가지수': np.random.uniform(1500, 3000, n_quarters),
    '수출증가율': np.random.uniform(-10, 20, n_quarters),
    '소비증가율': np.random.uniform(-5, 10, n_quarters)
})

# GDP 성장률 (합성 데이터)
economic_data['GDP성장률'] = (
    5 - 0.5 * economic_data['실업률'] +
    0.3 * economic_data['인플레이션'] -
    0.2 * economic_data['금리'] +
    0.001 * economic_data['주가지수'] +
    0.2 * economic_data['수출증가율'] +
    0.3 * economic_data['소비증가율'] +
    np.random.normal(0, 1, n_quarters)
)

print(economic_data.head())
print(f"\n평균 GDP 성장률: {economic_data['GDP성장률'].mean():.2f}%")
```

---

## 7.29 GDP 예측 모델

### Random Forest로 예측
```python
# 특성과 타겟
gdp_features = ['실업률', '인플레이션', '금리', '환율', 
                '주가지수', '수출증가율', '소비증가율']
X_gdp = economic_data[gdp_features]
y_gdp = economic_data['GDP성장률']

# 시계열이므로 시간 순서 유지하여 분할
split_idx = int(0.8 * len(X_gdp))
X_gdp_train = X_gdp[:split_idx]
X_gdp_test = X_gdp[split_idx:]
y_gdp_train = y_gdp[:split_idx]
y_gdp_test = y_gdp[split_idx:]

# Random Forest 학습
rf_gdp = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_gdp.fit(X_gdp_train, y_gdp_train)

# 예측
y_gdp_pred = rf_gdp.predict(X_gdp_test)

# 평가
mae_gdp = mean_absolute_error(y_gdp_test, y_gdp_pred)
r2_gdp = r2_score(y_gdp_test, y_gdp_pred)

print("===== GDP 성장률 예측 =====")
print(f"MAE: {mae_gdp:.2f}%p")
print(f"R²: {r2_gdp:.4f}")

# 변수 중요도
importance_gdp = pd.DataFrame({
    'Feature': gdp_features,
    'Importance': rf_gdp.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n변수 중요도:")
print(importance_gdp)
```

---

## 7.30 GDP 예측 시각화

### 실제 vs 예측 비교
```python
plt.figure(figsize=(12, 6))

# 실제값
plt.plot(range(split_idx, len(economic_data)), y_gdp_test, 
         'o-', label='실제 GDP 성장률', linewidth=2)

# 예측값
plt.plot(range(split_idx, len(economic_data)), y_gdp_pred, 
         's-', label='예측 GDP 성장률', linewidth=2, alpha=0.7)

plt.xlabel('분기')
plt.ylabel('GDP 성장률 (%)')
plt.title(f'GDP 성장률 예측 (MAE={mae_gdp:.2f}%p)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 예측 오차
errors = y_gdp_test - y_gdp_pred
plt.figure(figsize=(12, 4))
plt.bar(range(len(errors)), errors)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('테스트 분기')
plt.ylabel('예측 오차 (%p)')
plt.title('GDP 성장률 예측 오차')
plt.tight_layout()
plt.show()
```

---

## 7.31 ML 모델의 한계와 주의점

### 주의사항

#### 1. 인과관계 vs 상관관계
- ML은 상관관계만 학습
- 정책 평가는 인과관계 필요
- 해결: Causal ML 기법 활용

#### 2. 외삽 문제
- 학습 범위 밖 예측 불안정
- 특히 트리 기반 모델
- 해결: 학습 범위 확인

#### 3. 데이터 품질
- Garbage in, garbage out
- 데이터 편향이 모델에 반영
- 해결: 철저한 EDA

#### 4. 과적합
- 복잡한 모델일수록 위험
- 해결: 교차 검증, 정규화

---

## 7.32 경제학 연구에서 ML 활용

### 최근 연구 동향

#### 1. 노동경제학
- 임금 예측 및 격차 분석
- 구직 성공 확률 예측
- 경력 경로 분석

#### 2. 재무경제학
- 주가 예측
- 신용 위험 평가
- 포트폴리오 최적화

#### 3. 거시경제학
- GDP, 인플레이션 예측
- 경기순환 분류
- 정책 효과 예측

#### 4. 개발경제학
- 빈곤 예측 (위성 이미지 + ML)
- 원조 효과 분석
- 미시 데이터 분석

---

## 7.33 실전 팁: 모델 선택 가이드

### 상황별 권장 모델

#### 해석이 중요하면
→ **선형 회귀, Lasso**
- p-value, 계수 해석 명확

#### 예측이 중요하면
→ **Random Forest, XGBoost**
- 높은 정확도
- SHAP로 해석 보완

#### 인과관계 추정
→ **Double ML, Causal Forest**
- ML + 인과추론 결합

#### 변수 선택
→ **Lasso, Random Forest**
- 자동 변수 선택 기능

#### 대용량 데이터
→ **XGBoost, LightGBM**
- 빠른 속도

---

## 7.34 Python 라이브러리 정리

### 필수 라이브러리

#### Scikit-learn
- 표준 ML 라이브러리
- 다양한 모델, 전처리, 평가

#### XGBoost
- Gradient Boosting 최적화
- Kaggle 우승 단골

#### LightGBM
- Microsoft 개발
- 빠른 속도

#### SHAP
- 모델 해석
- 변수 기여도 시각화

#### EconML (Microsoft)
- 인과추론 + ML
- Double ML 구현

---

## 7.35 실습 프로젝트 아이디어

### 프로젝트 1: 부동산 가격 예측
- 데이터: 실거래가, 지역 특성
- 모델: Random Forest, XGBoost
- 목표: 정확한 가격 예측 + 주요 결정 요인

### 프로젝트 2: 신용 위험 평가
- 데이터: 개인 재무 정보, 신용 이력
- 모델: Logistic Regression, Random Forest
- 목표: 대출 승인 예측 + 공정성 분석

### 프로젝트 3: 주식 수익률 예측
- 데이터: 재무제표, 시장 지표
- 모델: LSTM, Random Forest
- 목표: 초과 수익 전략

---

## 7.36 Chapter 7 요약

### 배운 내용
✓ 계량경제학 vs 머신러닝 비교  
✓ Scikit-learn 기초  
✓ 선형 모델 (Ridge, Lasso)  
✓ 트리 모델 (Decision Tree, Random Forest)  
✓ Gradient Boosting (GB, XGBoost)  
✓ 변수 중요도 분석  
✓ SHAP values로 모델 해석  
✓ Double ML로 인과추론  
✓ 계량경제학 + ML 융합 방법

### 핵심 메시지
- ML은 예측력이 강함
- 해석 기법으로 블랙박스 극복
- 인과추론과 결합 시 강력

---

## 7.37 실습 과제

### 과제 1: 모델 비교
- Chapter 5의 임금 데이터 사용
- 5가지 모델 비교 (OLS, Ridge, Lasso, RF, XGBoost)
- 성능 평가 및 변수 중요도 분석

### 과제 2: 하이퍼파라미터 튜닝
- Random Forest Grid Search
- 최적 파라미터 찾기
- CV 점수와 Test 점수 비교

### 과제 3: SHAP 분석
- 최고 성능 모델 선택
- SHAP values 계산
- Summary plot, Waterfall plot 생성
- 주요 발견 해석

### 과제 4: 융합 방법
- Double ML 구현
- ML 변수선택 + OLS
- 결과 비교 및 해석

---

## 7.38 다음 단계

### Chapter 8 예고: 최신 연구와 나만의 실전 프로젝트
- 최신 계량경제학 연구 동향
- Causal ML, Deep Learning
- Kaggle 경진대회 소개
- 공공데이터 활용 프로젝트
- 포트폴리오 작성 가이드
- 연구/취업 준비

### 준비사항
- Chapter 1-7 전체 복습
- 관심 연구 주제 생각하기
- 데이터 출처 조사
- GitHub 계정 준비

---

## 7.39 참고 자료

### 교재
- "An Introduction to Statistical Learning" by James et al. (무료 PDF)
- "Hands-On Machine Learning" by Géron

### 온라인 강좌
- Coursera: Machine Learning by Andrew Ng
- Fast.ai: Practical Deep Learning

### 논문
- Athey & Imbens (2019): "Machine Learning Methods Economists Should Know About"
- Mullainathan & Spiess (2017): "Machine Learning: An Applied Econometric Approach"

### 라이브러리 문서
- Scikit-learn: https://scikit-learn.org/
- SHAP: https://shap.readthedocs.io/
- EconML: https://econml.azurewebsites.net/

---

## 7.40 추가 학습 자료

### Kaggle
- 실전 경진대회 참여
- 다른 사람의 코드 학습
- 데이터셋 탐색

### GitHub
- 오픈소스 프로젝트
- 코드 공유 및 협업

### 학회
- NBER (National Bureau of Economic Research)
- AEA (American Economic Association)
- ML for Economics Workshop

---

## Q&A

질문이 있으시면 편하게 물어보세요!

ML 모델 선택에 어려움이 있나요?
SHAP 해석이 궁금한가요?
실제 프로젝트 아이디어가 필요한가요?
