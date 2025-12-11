# Chapter 5: 실제 경제 데이터로 본 실전 분석

---

## 5.1 실전 분석 개요

### 이번 Chapter의 목표
- 실제 공개 경제 데이터 활용
- 전체 분석 파이프라인 실습
- 두 가지 실증 프로젝트 수행

### 실전 프로젝트
1. **임금 결정요인 분석**
   - 교육, 경력, 성별이 임금에 미치는 영향
2. **주택가격 결정요인 분석**
   - 지역, 면적, 시설 등이 가격에 미치는 영향

---

## 5.2 공개 경제 데이터 출처

### 한국 데이터
- **KOSIS (국가통계포털)**: https://kosis.kr/
- **공공데이터포털**: https://www.data.go.kr/
- **한국은행 경제통계**: https://ecos.bok.or.kr/
- **통계청 MDIS**: https://mdis.kostat.go.kr/

### 글로벌 데이터
- **World Bank Open Data**: https://data.worldbank.org/
- **OECD Statistics**: https://stats.oecd.org/
- **IMF Data**: https://www.imf.org/en/Data
- **Penn World Table**: https://www.rug.nl/ggdc/productivity/pwt/

---

## 5.3 데이터 선정 기준

### 좋은 데이터의 조건

#### 1. 신뢰성
- 공신력 있는 기관에서 제공
- 데이터 수집 방법 명시
- 정기적 업데이트

#### 2. 완결성
- 필요한 변수 포함
- 충분한 표본 크기
- 결측치 비율이 낮음

#### 3. 접근성
- 무료 또는 합리적 비용
- 다운로드 가능 형식 (CSV, Excel 등)
- 명확한 변수 설명 (코드북)

---

## 5.4 데이터 전처리 파이프라인

### 전처리 5단계

```python
# 1. 데이터 불러오기
df = pd.read_csv('data.csv')

# 2. 데이터 탐색 (EDA)
df.info()
df.describe()

# 3. 결측치 처리
df = df.dropna(subset=['key_variables'])

# 4. 이상치 처리
df = df[df['value'].between(Q1, Q3)]

# 5. 변수 변환
df['log_value'] = np.log(df['value'])
```

---

## 5.5 프로젝트 1: 임금 결정요인 분석

### 연구 질문
**"교육, 경력, 성별이 임금에 어떤 영향을 미치는가?"**

### 가설
- H1: 교육년수가 높을수록 임금이 높다
- H2: 경력이 많을수록 임금이 높다 (체감)
- H3: 성별 임금 격차가 존재한다

### 사용 데이터
- CPS (Current Population Survey) 스타일 데이터
- 또는 한국 노동패널 (KLIPS) 데이터

---

## 5.6 임금 데이터 생성

### 실습용 데이터 생성
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 2000

# 설명변수 생성
data = {
    '교육년수': np.random.choice(range(9, 21), n),
    '경력년수': np.random.uniform(0, 40, n),
    '성별': np.random.choice(['남성', '여성'], n),
    '지역': np.random.choice(['수도권', '광역시', '기타'], n),
    '산업': np.random.choice(['제조업', '서비스업', 'IT', '금융'], n),
    '결혼상태': np.random.choice(['미혼', '기혼'], n)
}

df_wage = pd.DataFrame(data)

# 더미변수 생성
df_wage['성별_더미'] = (df_wage['성별'] == '남성').astype(int)
df_wage['수도권_더미'] = (df_wage['지역'] == '수도권').astype(int)
df_wage['기혼_더미'] = (df_wage['결혼상태'] == '기혼').astype(int)
```

---

## 5.7 임금 데이터 생성 (계속)

### 임금 생성 (현실적 패턴 반영)
```python
# 로그 임금 생성 (Mincer equation 기반)
log_hourly_wage = (
    1.5 +                                      # 기본 임금
    0.08 * df_wage['교육년수'] +                # 교육 수익률 8%
    0.04 * df_wage['경력년수'] -                # 경력 효과
    0.0006 * (df_wage['경력년수'] ** 2) +      # 경력 체감
    0.15 * df_wage['성별_더미'] +               # 성별 격차 15%
    0.10 * df_wage['수도권_더미'] +             # 지역 프리미엄
    0.05 * df_wage['기혼_더미'] +               # 결혼 프리미엄
    np.random.normal(0, 0.3, n)               # 랜덤 오차
)

# 시간당 임금 (원화)
df_wage['시간당임금'] = np.exp(log_hourly_wage) * 10000
df_wage['log_임금'] = log_hourly_wage

# 월급 계산 (주 40시간 기준)
df_wage['월급'] = df_wage['시간당임금'] * 40 * 4.33

print(df_wage.head())
print(f"\n데이터 크기: {df_wage.shape}")
```

---

## 5.8 EDA: 기초 통계량

### 기본 통계
```python
# 수치형 변수 요약
print("===== 기초 통계량 =====")
print(df_wage[['교육년수', '경력년수', '시간당임금', '월급']].describe())

# 범주형 변수 분포
print("\n===== 범주형 변수 분포 =====")
print(df_wage['성별'].value_counts())
print(df_wage['지역'].value_counts())
print(df_wage['산업'].value_counts())
```

---

## 5.9 EDA: 임금 분포 시각화

### 임금 분포
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 시간당 임금 히스토그램
axes[0, 0].hist(df_wage['시간당임금'], bins=50, edgecolor='black')
axes[0, 0].set_xlabel('시간당 임금 (원)')
axes[0, 0].set_ylabel('빈도')
axes[0, 0].set_title('시간당 임금 분포')

# 2. 로그 임금 히스토그램
axes[0, 1].hist(df_wage['log_임금'], bins=50, edgecolor='black')
axes[0, 1].set_xlabel('log(시간당 임금)')
axes[0, 1].set_ylabel('빈도')
axes[0, 1].set_title('로그 임금 분포 (정규분포에 가까움)')

# 3. 성별 임금 박스플롯
df_wage.boxplot(column='시간당임금', by='성별', ax=axes[1, 0])
axes[1, 0].set_title('성별 임금 분포')
axes[1, 0].set_xlabel('성별')
axes[1, 0].set_ylabel('시간당 임금 (원)')

# 4. 지역별 임금 박스플롯
df_wage.boxplot(column='시간당임금', by='지역', ax=axes[1, 1])
axes[1, 1].set_title('지역별 임금 분포')
axes[1, 1].set_xlabel('지역')
axes[1, 1].set_ylabel('시간당 임금 (원)')

plt.tight_layout()
plt.show()
```

---

## 5.10 EDA: 교육과 경력의 효과

### 산점도
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. 교육년수 vs 임금
axes[0].scatter(df_wage['교육년수'], df_wage['시간당임금'], alpha=0.3)
axes[0].set_xlabel('교육년수')
axes[0].set_ylabel('시간당 임금 (원)')
axes[0].set_title('교육년수와 임금의 관계')

# 2. 경력년수 vs 임금
axes[1].scatter(df_wage['경력년수'], df_wage['시간당임금'], alpha=0.3)
axes[1].set_xlabel('경력년수')
axes[1].set_ylabel('시간당 임금 (원)')
axes[1].set_title('경력년수와 임금의 관계 (비선형)')

plt.tight_layout()
plt.show()
```

---

## 5.11 EDA: 그룹별 평균 임금

### 그룹 통계
```python
# 성별 평균 임금
gender_wage = df_wage.groupby('성별')['시간당임금'].agg(['mean', 'median', 'std'])
print("===== 성별 임금 통계 =====")
print(gender_wage)

# 지역별 평균 임금
region_wage = df_wage.groupby('지역')['시간당임금'].agg(['mean', 'median', 'std'])
print("\n===== 지역별 임금 통계 =====")
print(region_wage)

# 산업별 평균 임금
industry_wage = df_wage.groupby('산업')['시간당임금'].agg(['mean', 'median', 'std'])
print("\n===== 산업별 임금 통계 =====")
print(industry_wage)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

gender_wage['mean'].plot(kind='bar', ax=axes[0], color=['pink', 'lightblue'])
axes[0].set_title('성별 평균 임금')
axes[0].set_ylabel('시간당 임금 (원)')

region_wage['mean'].plot(kind='bar', ax=axes[1])
axes[1].set_title('지역별 평균 임금')
axes[1].set_ylabel('시간당 임금 (원)')

industry_wage['mean'].plot(kind='bar', ax=axes[2])
axes[2].set_title('산업별 평균 임금')
axes[2].set_ylabel('시간당 임금 (원)')

plt.tight_layout()
plt.show()
```

---

## 5.12 모델 1: 기본 임금 방정식

### Mincer Equation
```python
# 모델 1: 교육과 경력만
X1 = df_wage[['교육년수', '경력년수']]
X1 = sm.add_constant(X1)
y = df_wage['log_임금']

model1 = sm.OLS(y, X1).fit()
print("===== Model 1: 기본 모델 =====")
print(model1.summary())
```

### 해석
```python
print(f"\n교육 수익률: {model1.params['교육년수']*100:.2f}%")
print(f"해석: 교육 1년 증가 → 임금 {model1.params['교육년수']*100:.2f}% 증가")

print(f"\n경력 효과: {model1.params['경력년수']*100:.2f}%")
print(f"해석: 경력 1년 증가 → 임금 {model1.params['경력년수']*100:.2f}% 증가")
```

---

## 5.13 모델 2: 경력의 비선형 효과

### 경력 제곱항 추가
```python
# 경력 제곱 변수 생성
df_wage['경력년수_제곱'] = df_wage['경력년수'] ** 2

# 모델 2: 경력 제곱항 포함
X2 = df_wage[['교육년수', '경력년수', '경력년수_제곱']]
X2 = sm.add_constant(X2)

model2 = sm.OLS(y, X2).fit()
print("===== Model 2: 경력 제곱항 포함 =====")
print(model2.summary())
```

### 최적 경력 계산
```python
# 임금을 최대화하는 경력년수
optimal_exp = -model2.params['경력년수'] / (2 * model2.params['경력년수_제곱'])
print(f"\n임금 최대화 경력년수: {optimal_exp:.1f}년")
```

---

## 5.14 모델 3: 성별과 지역 효과

### 더미변수 추가
```python
# 모델 3: 성별, 지역 포함
X3 = df_wage[['교육년수', '경력년수', '경력년수_제곱', 
              '성별_더미', '수도권_더미', '기혼_더미']]
X3 = sm.add_constant(X3)

model3 = sm.OLS(y, X3).fit()
print("===== Model 3: 완전 모델 =====")
print(model3.summary())
```

### 성별 임금 격차 계산
```python
gender_gap = (np.exp(model3.params['성별_더미']) - 1) * 100
print(f"\n성별 임금 격차: {gender_gap:.2f}%")
print(f"해석: 다른 조건이 같을 때, 남성이 여성보다 {gender_gap:.2f}% 더 높은 임금")
```

---

## 5.15 모델 비교

### R-squared 비교
```python
print("===== 모델 성능 비교 =====")
print(f"Model 1 R²: {model1.rsquared:.4f}")
print(f"Model 2 R²: {model2.rsquared:.4f}")
print(f"Model 3 R²: {model3.rsquared:.4f}")

print(f"\nModel 1 Adj R²: {model1.rsquared_adj:.4f}")
print(f"Model 2 Adj R²: {model2.rsquared_adj:.4f}")
print(f"Model 3 Adj R²: {model3.rsquared_adj:.4f}")
```

### 시각화
```python
models = ['Model 1', 'Model 2', 'Model 3']
r2_values = [model1.rsquared, model2.rsquared, model3.rsquared]
adj_r2_values = [model1.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, r2_values, width, label='R²')
ax.bar(x + width/2, adj_r2_values, width, label='Adjusted R²')

ax.set_xlabel('모델')
ax.set_ylabel('R²')
ax.set_title('모델별 설명력 비교')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
plt.show()
```

---

## 5.16 회귀진단

### 잔차 분석
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 잔차 vs 적합값
axes[0, 0].scatter(model3.fittedvalues, model3.resid, alpha=0.3)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('적합값')
axes[0, 0].set_ylabel('잔차')
axes[0, 0].set_title('잔차 플롯')

# 2. Q-Q plot
sm.qqplot(model3.resid, line='45', ax=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# 3. 잔차 히스토그램
axes[1, 0].hist(model3.resid, bins=50, edgecolor='black')
axes[1, 0].set_xlabel('잔차')
axes[1, 0].set_ylabel('빈도')
axes[1, 0].set_title('잔차 분포')

# 4. Scale-Location
axes[1, 1].scatter(model3.fittedvalues, np.sqrt(np.abs(model3.resid)), alpha=0.3)
axes[1, 1].set_xlabel('적합값')
axes[1, 1].set_ylabel('√|잔차|')
axes[1, 1].set_title('Scale-Location Plot')

plt.tight_layout()
plt.show()
```

---

## 5.17 VIF 확인

### 다중공선성 진단
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF 계산
X_vif = df_wage[['교육년수', '경력년수', '경력년수_제곱', 
                 '성별_더미', '수도권_더미', '기혼_더미']]

vif_data = pd.DataFrame()
vif_data['Variable'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) 
                   for i in range(X_vif.shape[1])]

print("===== VIF 분석 =====")
print(vif_data)
print("\n해석: VIF > 10이면 심각한 다중공선성")
```

---

## 5.18 예측 예시

### 특정 프로필 임금 예측
```python
# 예측할 사람들의 프로필
profiles = pd.DataFrame({
    'const': [1, 1, 1],
    '교육년수': [16, 12, 18],
    '경력년수': [10, 20, 5],
    '경력년수_제곱': [100, 400, 25],
    '성별_더미': [1, 0, 1],  # 남성, 여성, 남성
    '수도권_더미': [1, 0, 1],
    '기혼_더미': [1, 1, 0]
}, index=['Profile A', 'Profile B', 'Profile C'])

# 로그 임금 예측
log_wage_pred = model3.predict(profiles)

# 실제 임금으로 변환
wage_pred = np.exp(log_wage_pred) * 10000

print("===== 임금 예측 =====")
for profile, wage in zip(profiles.index, wage_pred):
    print(f"{profile}: {wage:,.0f}원/시간 (월급 약 {wage*40*4.33:,.0f}원)")
```

---

## 5.19 프로젝트 2: 주택가격 결정요인

### 연구 질문
**"주택의 특성과 위치가 가격에 어떤 영향을 미치는가?"**

### 가설
- H1: 면적이 클수록 가격이 높다
- H2: 역세권일수록 가격이 높다
- H3: 건축년도가 최근일수록 가격이 높다
- H4: 학군이 좋을수록 가격이 높다

### 데이터 유형
- Boston Housing Dataset 스타일
- 또는 실제 부동산 거래 데이터

---

## 5.20 주택 데이터 생성

### 실습용 데이터
```python
np.random.seed(123)
n_houses = 1500

# 주택 특성
housing_data = {
    '면적': np.random.uniform(40, 200, n_houses),  # m²
    '방개수': np.random.choice([1, 2, 3, 4, 5], n_houses),
    '욕실개수': np.random.choice([1, 2, 3], n_houses),
    '건축년도': np.random.randint(1980, 2024, n_houses),
    '층수': np.random.randint(1, 30, n_houses),
    '역까지거리': np.random.uniform(0.1, 2.0, n_houses),  # km
    '학교까지거리': np.random.uniform(0.2, 3.0, n_houses),  # km
    '공원유무': np.random.choice([0, 1], n_houses),
    '주차가능대수': np.random.randint(0, 3, n_houses),
    '지역': np.random.choice(['강남', '서초', '송파', '기타'], n_houses)
}

df_house = pd.DataFrame(housing_data)

# 더미변수
df_house['강남_더미'] = (df_house['지역'] == '강남').astype(int)
df_house['서초_더미'] = (df_house['지역'] == '서초').astype(int)
df_house['송파_더미'] = (df_house['지역'] == '송파').astype(int)
```

---

## 5.21 주택 가격 생성

### 가격 결정 모델
```python
# 로그 가격 생성 (현실적 패턴)
log_price = (
    10.0 +                                      # 기본 가격
    0.015 * df_house['면적'] +                  # 면적 효과
    0.08 * df_house['방개수'] +                 # 방 개수
    0.05 * df_house['욕실개수'] +               # 욕실
    0.01 * (df_house['건축년도'] - 1980) +      # 건축년도
    -0.15 * df_house['역까지거리'] +            # 역세권 (거리 증가 → 가격 하락)
    -0.08 * df_house['학교까지거리'] +          # 학군
    0.10 * df_house['공원유무'] +               # 공원 근처
    0.05 * df_house['주차가능대수'] +           # 주차
    0.30 * df_house['강남_더미'] +              # 강남 프리미엄
    0.25 * df_house['서초_더미'] +              # 서초 프리미엄
    0.20 * df_house['송파_더미'] +              # 송파 프리미엄
    np.random.normal(0, 0.2, n_houses)         # 랜덤 오차
)

# 실제 가격 (억 원)
df_house['가격'] = np.exp(log_price) / 10
df_house['log_가격'] = log_price

print(df_house.head(10))
print(f"\n데이터 크기: {df_house.shape}")
print(f"평균 가격: {df_house['가격'].mean():.2f}억 원")
print(f"가격 범위: {df_house['가격'].min():.2f} ~ {df_house['가격'].max():.2f}억 원")
```

---

## 5.22 주택 데이터 EDA

### 기초 통계
```python
# 수치형 변수 요약
print("===== 주택 데이터 기초 통계 =====")
print(df_house[['면적', '방개수', '건축년도', '가격']].describe())

# 가격 분포
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_house['가격'], bins=50, edgecolor='black')
axes[0].set_xlabel('가격 (억 원)')
axes[0].set_ylabel('빈도')
axes[0].set_title('주택 가격 분포')

axes[1].hist(df_house['log_가격'], bins=50, edgecolor='black')
axes[1].set_xlabel('log(가격)')
axes[1].set_ylabel('빈도')
axes[1].set_title('로그 가격 분포')

plt.tight_layout()
plt.show()
```

---

## 5.23 주택 데이터 시각화

### 주요 변수 관계
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 면적 vs 가격
axes[0, 0].scatter(df_house['면적'], df_house['가격'], alpha=0.3)
axes[0, 0].set_xlabel('면적 (m²)')
axes[0, 0].set_ylabel('가격 (억 원)')
axes[0, 0].set_title('면적과 가격의 관계')

# 2. 건축년도 vs 가격
axes[0, 1].scatter(df_house['건축년도'], df_house['가격'], alpha=0.3)
axes[0, 1].set_xlabel('건축년도')
axes[0, 1].set_ylabel('가격 (억 원)')
axes[0, 1].set_title('건축년도와 가격의 관계')

# 3. 역까지 거리 vs 가격
axes[1, 0].scatter(df_house['역까지거리'], df_house['가격'], alpha=0.3)
axes[1, 0].set_xlabel('역까지 거리 (km)')
axes[1, 0].set_ylabel('가격 (억 원)')
axes[1, 0].set_title('역세권과 가격의 관계')

# 4. 지역별 가격 박스플롯
df_house.boxplot(column='가격', by='지역', ax=axes[1, 1])
axes[1, 1].set_title('지역별 가격 분포')
axes[1, 1].set_xlabel('지역')
axes[1, 1].set_ylabel('가격 (억 원)')

plt.tight_layout()
plt.show()
```

---

## 5.24 지역별 평균 가격

### 그룹 통계
```python
# 지역별 통계
region_stats = df_house.groupby('지역')['가격'].agg(['mean', 'median', 'std', 'count'])
print("===== 지역별 가격 통계 =====")
print(region_stats)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

region_stats['mean'].plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('지역별 평균 가격')
axes[0].set_ylabel('가격 (억 원)')
axes[0].set_xlabel('지역')

# 방 개수별 평균 가격
room_stats = df_house.groupby('방개수')['가격'].mean()
room_stats.plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('방 개수별 평균 가격')
axes[1].set_ylabel('가격 (억 원)')
axes[1].set_xlabel('방 개수')

plt.tight_layout()
plt.show()
```

---

## 5.25 상관관계 분석

### 상관계수 히트맵
```python
# 수치형 변수 선택
numeric_cols = ['면적', '방개수', '욕실개수', '건축년도', '층수', 
                '역까지거리', '학교까지거리', '주차가능대수', '가격']

# 상관계수 행렬
corr_matrix = df_house[numeric_cols].corr()

# 히트맵
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True)
plt.title('변수 간 상관계수')
plt.tight_layout()
plt.show()

# 가격과의 상관계수 정렬
price_corr = corr_matrix['가격'].sort_values(ascending=False)
print("===== 가격과의 상관계수 =====")
print(price_corr)
```

---

## 5.26 주택가격 회귀분석 - Model 1

### 기본 모델
```python
# Model 1: 주택 특성만
X_h1 = df_house[['면적', '방개수', '욕실개수', '건축년도']]
X_h1 = sm.add_constant(X_h1)
y_h = df_house['log_가격']

house_model1 = sm.OLS(y_h, X_h1).fit()
print("===== 주택가격 Model 1 =====")
print(house_model1.summary())
```

### 해석
```python
print(f"\n면적 계수: {house_model1.params['면적']:.4f}")
print(f"해석: 면적 1m² 증가 → 가격 {house_model1.params['면적']*100:.2f}% 증가")

print(f"\n건축년도 계수: {house_model1.params['건축년도']:.4f}")
print(f"해석: 건축년도 1년 최근 → 가격 {house_model1.params['건축년도']*100:.2f}% 증가")
```

---

## 5.27 주택가격 회귀분석 - Model 2

### 위치 변수 추가
```python
# Model 2: 위치 변수 추가
X_h2 = df_house[['면적', '방개수', '욕실개수', '건축년도',
                 '역까지거리', '학교까지거리', '공원유무', '주차가능대수']]
X_h2 = sm.add_constant(X_h2)

house_model2 = sm.OLS(y_h, X_h2).fit()
print("===== 주택가격 Model 2 =====")
print(house_model2.summary())
```

### 역세권 효과
```python
station_effect = house_model2.params['역까지거리']
print(f"\n역세권 효과: {station_effect*100:.2f}%")
print(f"해석: 역까지 거리 1km 증가 → 가격 {abs(station_effect)*100:.2f}% 하락")
```

---

## 5.28 주택가격 회귀분석 - Model 3

### 완전 모델 (지역 더미 포함)
```python
# Model 3: 지역 더미 추가
X_h3 = df_house[['면적', '방개수', '욕실개수', '건축년도',
                 '역까지거리', '학교까지거리', '공원유무', '주차가능대수',
                 '강남_더미', '서초_더미', '송파_더미']]
X_h3 = sm.add_constant(X_h3)

house_model3 = sm.OLS(y_h, X_h3).fit()
print("===== 주택가격 Model 3: 완전 모델 =====")
print(house_model3.summary())
```

---

## 5.29 지역 프리미엄 계산

### 지역별 가격 프리미엄
```python
print("===== 지역 프리미엄 (기타 지역 대비) =====")

regions = ['강남', '서초', '송파']
dummies = ['강남_더미', '서초_더미', '송파_더미']

for region, dummy in zip(regions, dummies):
    premium = (np.exp(house_model3.params[dummy]) - 1) * 100
    print(f"{region}: {premium:.2f}% 높음")

# 시각화
premiums = [(np.exp(house_model3.params[d]) - 1) * 100 for d in dummies]

plt.figure(figsize=(10, 6))
plt.bar(regions, premiums, color=['gold', 'silver', 'bronze'])
plt.ylabel('가격 프리미엄 (%)')
plt.title('지역별 가격 프리미엄 (기타 지역 대비)')
plt.axhline(y=0, color='black', linestyle='--')
for i, v in enumerate(premiums):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
plt.show()
```

---

## 5.30 모델 성능 비교

### R-squared 비교
```python
print("===== 주택가격 모델 비교 =====")
models_names = ['Model 1\n(주택특성)', 'Model 2\n(+위치)', 'Model 3\n(+지역)']
r2_vals = [house_model1.rsquared, house_model2.rsquared, house_model3.rsquared]
adj_r2_vals = [house_model1.rsquared_adj, house_model2.rsquared_adj, house_model3.rsquared_adj]

print(f"Model 1 R²: {house_model1.rsquared:.4f}")
print(f"Model 2 R²: {house_model2.rsquared:.4f}")
print(f"Model 3 R²: {house_model3.rsquared:.4f}")

# 시각화
x = np.arange(len(models_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, r2_vals, width, label='R²', alpha=0.8)
ax.bar(x + width/2, adj_r2_vals, width, label='Adjusted R²', alpha=0.8)

ax.set_ylabel('R²')
ax.set_title('주택가격 모델 성능 비교')
ax.set_xticks(x)
ax.set_xticklabels(models_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for i, (r2, adj_r2) in enumerate(zip(r2_vals, adj_r2_vals)):
    ax.text(i - width/2, r2 + 0.01, f'{r2:.3f}', ha='center', va='bottom')
    ax.text(i + width/2, adj_r2 + 0.01, f'{adj_r2:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

---

## 5.31 주택가격 예측 예시

### 특정 주택 가격 예측
```python
# 예측할 주택들
sample_houses = pd.DataFrame({
    'const': [1, 1, 1],
    '면적': [85, 130, 60],
    '방개수': [3, 4, 2],
    '욕실개수': [2, 2, 1],
    '건축년도': [2020, 2015, 2005],
    '역까지거리': [0.3, 0.8, 1.5],
    '학교까지거리': [0.5, 1.0, 2.0],
    '공원유무': [1, 1, 0],
    '주차가능대수': [2, 2, 1],
    '강남_더미': [1, 0, 0],
    '서초_더미': [0, 1, 0],
    '송파_더미': [0, 0, 0]
}, index=['House A (강남)', 'House B (서초)', 'House C (기타)'])

# 로그 가격 예측
log_price_pred = house_model3.predict(sample_houses)

# 실제 가격으로 변환
price_pred = np.exp(log_price_pred) / 10

print("===== 주택 가격 예측 =====")
for house, price in zip(sample_houses.index, price_pred):
    print(f"{house}: {price:.2f}억 원")

# 상세 정보와 함께 출력
result_df = sample_houses[['면적', '방개수', '건축년도', '역까지거리']].copy()
result_df['예측가격(억)'] = price_pred.values

print("\n===== 예측 결과 상세 =====")
print(result_df)
```

---

## 5.32 회귀진단 - 주택가격 모델

### 잔차 분석
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 잔차 vs 적합값
axes[0, 0].scatter(house_model3.fittedvalues, house_model3.resid, alpha=0.3)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('적합값')
axes[0, 0].set_ylabel('잔차')
axes[0, 0].set_title('잔차 플롯')

# 2. Q-Q plot
sm.qqplot(house_model3.resid, line='45', ax=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# 3. 잔차 히스토그램
axes[1, 0].hist(house_model3.resid, bins=50, edgecolor='black')
axes[1, 0].set_xlabel('잔차')
axes[1, 0].set_ylabel('빈도')
axes[1, 0].set_title('잔차 분포')

# 4. 실제 vs 예측
axes[1, 1].scatter(y_h, house_model3.fittedvalues, alpha=0.3)
axes[1, 1].plot([y_h.min(), y_h.max()], [y_h.min(), y_h.max()], 
                'r--', linewidth=2)
axes[1, 1].set_xlabel('실제 log(가격)')
axes[1, 1].set_ylabel('예측 log(가격)')
axes[1, 1].set_title('실제 vs 예측')

plt.tight_layout()
plt.show()
```

---

## 5.33 Feature Importance 분석

### 표준화 계수 비교
```python
# 표준화 계수 계산 (비교 가능하도록)
from sklearn.preprocessing import StandardScaler

X_h3_nostd = df_house[['면적', '방개수', '욕실개수', '건축년도',
                        '역까지거리', '학교까지거리', '공원유무', '주차가능대수',
                        '강남_더미', '서초_더미', '송파_더미']]

scaler = StandardScaler()
X_h3_std = scaler.fit_transform(X_h3_nostd)
X_h3_std = sm.add_constant(X_h3_std)

# 표준화된 데이터로 회귀
house_model_std = sm.OLS(y_h, X_h3_std).fit()

# 계수 절대값 기준 정렬
coef_df = pd.DataFrame({
    'Variable': X_h3_nostd.columns,
    'Std_Coef': house_model_std.params[1:]  # 상수항 제외
})
coef_df['Abs_Coef'] = coef_df['Std_Coef'].abs()
coef_df = coef_df.sort_values('Abs_Coef', ascending=False)

print("===== 변수 중요도 (표준화 계수) =====")
print(coef_df)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Variable'], coef_df['Std_Coef'])
plt.xlabel('표준화 계수')
plt.title('주택가격 결정요인 중요도')
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.show()
```

---

## 5.34 AI 추천: 예측 개선 방법

### 1. 변수 상호작용
```python
# 예: 면적 × 강남_더미 (강남에서 면적 효과가 더 클 수 있음)
df_house['면적_강남'] = df_house['면적'] * df_house['강남_더미']

X_h4 = df_house[['면적', '방개수', '욕실개수', '건축년도',
                 '역까지거리', '학교까지거리', '공원유무', '주차가능대수',
                 '강남_더미', '서초_더미', '송파_더미', '면적_강남']]
X_h4 = sm.add_constant(X_h4)

house_model4 = sm.OLS(y_h, X_h4).fit()
print("===== Model 4: 상호작용 포함 =====")
print(f"R²: {house_model4.rsquared:.4f}")
print(f"Adj R²: {house_model4.rsquared_adj:.4f}")
```

---

## 5.35 AI 추천: 비선형 관계

### 2. 다항 회귀
```python
# 면적의 제곱항 추가 (체감 효과)
df_house['면적_제곱'] = df_house['면적'] ** 2

X_h5 = df_house[['면적', '면적_제곱', '방개수', '욕실개수', '건축년도',
                 '역까지거리', '학교까지거리', '공원유무', '주차가능대수',
                 '강남_더미', '서초_더미', '송파_더미']]
X_h5 = sm.add_constant(X_h5)

house_model5 = sm.OLS(y_h, X_h5).fit()
print("===== Model 5: 면적 제곱항 포함 =====")
print(f"R²: {house_model5.rsquared:.4f}")
print(f"Adj R²: {house_model5.rsquared_adj:.4f}")
```

---

## 5.36 실전 팁: 데이터 수집

### 실제 데이터 출처

#### 임금 데이터
- **한국노동패널(KLIPS)**: https://www.kli.re.kr/klips/
- **고용형태별근로실태조사**: 통계청
- **임금구조 기본통계조사**: 고용노동부

#### 주택 데이터
- **부동산 실거래가**: 국토교통부
- **KB부동산 시세**: https://onland.kbstar.com/
- **아파트 실거래가**: 공공데이터포털

---

## 5.37 실전 팁: 변수 선택

### 변수 선택 기준

#### 1. 이론적 근거
- 경제학 이론에 기반
- 선행 연구 참고

#### 2. 통계적 유의성
- p-value < 0.05
- 신뢰구간이 0을 포함하지 않음

#### 3. 경제적 유의성
- 계수의 크기가 의미있는가?
- 실질적 영향이 큰가?

#### 4. 다중공선성 고려
- VIF < 10
- 상관계수가 너무 높지 않음

---

## 5.38 실전 팁: 보고서 작성

### 분석 보고서 구조

#### 1. 서론
- 연구 질문과 배경
- 데이터 출처와 기간
- 분석 방법 개요

#### 2. 데이터 및 방법론
- 변수 설명 (기초통계)
- 모델 설정
- 추정 방법

#### 3. 분석 결과
- 회귀분석 결과표
- 주요 계수 해석
- 회귀진단 결과

#### 4. 결론
- 주요 발견
- 정책적 시사점
- 한계점 및 향후 연구

---

## 5.39 실전 팁: 결과 시각화

### 효과적인 시각화
```python
# 예: 교육 수익률 시각화
education_returns = []
education_levels = range(9, 21)

for edu in education_levels:
    pred_wage = (model3.params['const'] + 
                 model3.params['교육년수'] * edu +
                 model3.params['경력년수'] * 10 +
                 model3.params['경력년수_제곱'] * 100)
    education_returns.append(np.exp(pred_wage) * 10000)

plt.figure(figsize=(10, 6))
plt.plot(education_levels, education_returns, 'o-', linewidth=2, markersize=8)
plt.xlabel('교육년수', fontsize=12)
plt.ylabel('예상 시간당 임금 (원)', fontsize=12)
plt.title('교육년수별 예상 임금 (경력 10년 기준)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 5.40 Chapter 5 요약

### 배운 내용
✓ 실제 경제 데이터 출처 및 수집  
✓ 데이터 전처리 파이프라인  
✓ 임금 결정요인 실증분석 (Mincer equation)  
✓ 주택가격 결정요인 분석  
✓ 모델 비교 및 선택  
✓ 예측 및 해석  
✓ 분석 보고서 작성 방법

### 핵심 교훈
- 이론 → 데이터 → 모델 → 해석
- 여러 모델 비교하여 최선 선택
- 회귀진단은 필수
- 경제적 해석이 중요

---

## 5.41 실습 과제

### 과제 1: 임금 분석 확장
- 산업 더미변수 추가
- 교육과 경력의 상호작용 효과 분석
- 성별 임금 격차를 산업별로 비교

### 과제 2: 주택가격 분석 확장
- 층수의 비선형 효과 탐색
- 면적과 지역의 상호작용 분석
- 건축년도를 범주형으로 변환하여 분석

### 과제 3: 실제 데이터 프로젝트
- 공개 데이터 다운로드 (KOSIS, 공공데이터포털)
- 자신만의 연구 질문 설정
- 전체 분석 수행 및 보고서 작성

---

## 5.42 다음 단계

### Chapter 6 예고: 모델 선택과 심화 분석
- 선형 vs 비선형 모델 비교
- 패널 vs 횡단면 데이터 선택
- 강건성 테스트 (robustness checks)
- 예측력 평가 지표
- 경제적 해석력 높이기

### 준비사항
- Chapter 5의 두 프로젝트 복습
- 모델 평가 지표 예습
- 과적합(overfitting) 개념 이해

---

## 참고 자료

### 데이터 출처 재정리
- **KOSIS**: https://kosis.kr/
- **공공데이터포털**: https://www.data.go.kr/
- **한국은행**: https://ecos.bok.or.kr/
- **World Bank**: https://data.worldbank.org/

### 추천 논문
- Mincer, J. (1974). "Schooling, Experience, and Earnings"
- Rosen, S. (1974). "Hedonic Prices and Implicit Markets"

### Python 라이브러리
- `pandas`: 데이터 처리
- `statsmodels`: 회귀분석
- `matplotlib/seaborn`: 시각화

---

## Q&A

질문이 있으시면 편하게 물어보세요!

분석 결과를 해석하는 데 어려움이 있나요?
실제 데이터를 어디서 구해야 할지 궁금한가요?
