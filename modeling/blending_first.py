import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# # 전처리 함수 불러오기
# from preprocessing import load_and_preprocess_data

# # 데이터 로드
# X_train, X_test, y_train, y_test = load_and_preprocess_data()

# # 전처리 함수 불러오기
# from preprocessing import load_and_preprocess_data

# # 데이터 로드
# X_train, X_test, y_train, y_test = load_and_preprocess_data()


import pandas as pd
# 전처리 함수 불러오기
from preprocessing import load_and_preprocess_data

print("전처리된 데이터를 불러옵니다...")
X_train = pd.read_csv('train_features.csv')
y_train = pd.read_csv('train_target.csv').squeeze() # squeeze()로 Series 변환
X_test = pd.read_csv('test_features.csv')
y_test = pd.read_csv('test_target.csv').squeeze()

# --- 블렌딩 모델 구현 ---
print("\n--- 블렌딩 모델 학습 및 평가 (ROC-AUC Score, AUPRC) ---")

# 1. 학습 데이터 분할
X_train_base, X_holdout, y_train_base, y_holdout = train_test_split(
    X_train, y_train, test_size=0.5, random_state=42, stratify=y_train
)

# 2. 기본 모델 학습
lgbm_base = LGBMClassifier(random_state=42)
rf_base = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

lgbm_base.fit(X_train_base, y_train_base)
rf_base.fit(X_train_base, y_train_base)

# 3. 메타 모델 학습 데이터 생성
holdout_preds_lgbm = lgbm_base.predict_proba(X_holdout)[:, 1]
holdout_preds_rf = rf_base.predict_proba(X_holdout)[:, 1]
meta_X_train = np.c_[holdout_preds_lgbm, holdout_preds_rf]

# 4. 메타 모델 학습
meta_model = LogisticRegression()
meta_model.fit(meta_X_train, y_holdout)

# 5. 최종 예측
test_preds_lgbm = lgbm_base.predict_proba(X_test)[:, 1]
test_preds_rf = rf_base.predict_proba(X_test)[:, 1]
meta_X_test = np.c_[test_preds_lgbm, test_preds_rf]

blend_proba = meta_model.predict_proba(meta_X_test)[:, 1]  # Positive 클래스(1)에 대한 확률
blend_preds = (blend_proba > 0.5).astype(int)

# 평가
print("블렌딩 모델 평가 결과:")
print(classification_report(y_test, blend_preds, digits=4))
print(f"ROC-AUC Score: {roc_auc_score(y_test, blend_proba):.4f}")
print(f"AUPRC: {average_precision_score(y_test, blend_proba):.4f}")
