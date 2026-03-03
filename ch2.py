# %%
## 2 실습: 경제 데이터 EDA
### 실습 데이터 생성
import numpy as np
import pandas as pd

np.random.seed(42)

# 1000명의 가구 데이터 생성
n = 1000
data = {
    "가구ID": range(1, n + 1),
    "소득": np.random.lognormal(10, 0.5, n),
    "소비": np.random.lognormal(9, 0.6, n),
    "가구원수": np.random.choice([1, 2, 3, 4, 5], n),
    "지역": np.random.choice(["서울", "경기", "부산", "기타"], n),
    "교육년수": np.random.normal(14, 3, n),
}

df = pd.DataFrame(data)

# 일부 결측치 생성
df.loc[np.random.choice(df.index, 50), "소득"] = np.nan
df.loc[np.random.choice(df.index, 30), "교육년수"] = np.nan

# %%
# 데이터 구조
print(df.shape)
print(df.info())

# %%
# 처음 몇 행
print(df.head(10))

# 기초 통계
print(df.describe())

# %%
# 결측치 확인
print(df.isnull().sum())

# 결측치 비율
print(df.isnull().mean() * 100)

# %%
# 소득: 중앙값으로 대체
# 교육년수: 평균으로 대체
df = df.fillna({"소득": df["소득"].median(), "교육년수": df["교육년수"].mean()})

# 확인
print(df.isnull().sum())

# %%

### Step 3: 이상치 탐지 및 처리
# 소득 이상치 확인
Q1 = df["소득"].quantile(0.25)
Q3 = df["소득"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 이상치 개수
outliers = df[(df["소득"] < lower) | (df["소득"] > upper)]
print(f"이상치 개수: {len(outliers)}")

# 이상치 제거
df_clean = df[(df["소득"] >= lower) & (df["소득"] <= upper)]
print(f"정제 후 데이터: {df_clean.shape}")

# %%

### Step 4: 데이터 시각화
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Mac)
mpl.rcParams["font.family"] = "AppleGothic"

# 마이너스 깨짐 방지
mpl.rcParams["axes.unicode_minus"] = False

# 소득 분포
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_clean["소득"], bins=30)
axes[0].set_title("소득 분포")
axes[0].set_xlabel("소득")

axes[1].boxplot(df_clean["소득"])
axes[1].set_title("소득 박스플롯")
axes[1].set_ylabel("소득")

plt.tight_layout()
plt.show()

# %%

### Step 5: 변수 간 관계
# 수치형 변수만 선택
numeric_cols = ["소득", "소비", "가구원수", "교육년수"]
df_numeric = df_clean[numeric_cols]

# 상관계수 행렬
corr_matrix = df_numeric.corr()
print(corr_matrix)

# 히트맵
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("상관계수 행렬")
plt.show()

# %%

### Step 6: 범주형 변수 분석
# 지역별 평균 소득
region_income = df_clean.groupby("지역")["소득"].mean()
print(region_income)

# 시각화
region_income.plot(kind="bar")
plt.title("지역별 평균 소득")
plt.xlabel("지역")
plt.ylabel("평균 소득")
plt.show()

# %%

# 가구원수별 평균 소비
household_consumption = df_clean.groupby("가구원수")["소비"].mean()
print(household_consumption)

# %%
## 데이터 변환 기법

### 로그 변환
# 소득의 분포가 치우쳐있을 때
df_clean = df[df["소득"] > 0].copy()
df_clean["소득_log"] = np.log(df_clean["소득"])

# 변환 전후 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_clean["소득"], bins=30)
axes[0].set_title("원본 소득")

axes[1].hist(df_clean["소득_log"], bins=30)
axes[1].set_title("로그 변환 소득")

plt.tight_layout()
plt.show()

# %%

## 데이터 정규화/표준화

### 표준화 (Standardization)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_clean["소득_표준화"] = scaler.fit_transform(df_clean[["소득"]])

# 평균 0, 표준편차 1
print(df_clean["소득_표준화"].mean())  # ~0
print(df_clean["소득_표준화"].std())  # ~1

# %%

### 정규화 (Normalization)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_clean["소득_정규화"] = scaler.fit_transform(df_clean[["소득"]])

# 0과 1 사이 값
print(df_clean["소득_정규화"].min())  # 0
print(df_clean["소득_정규화"].max())  # 1
