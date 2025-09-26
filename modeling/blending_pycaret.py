import pandas as pd
import numpy as np
import pickle
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# --------------------------------------------------------------------------
# 1. 전체 학습 데이터 준비
# --------------------------------------------------------------------------
print("--- 전처리된 데이터 불러오기 시작 ---")
X_train_full = pd.read_csv('train_features.csv')
y_train_full = pd.read_csv('train_target.csv').squeeze()
X_test = pd.read_csv('test_features.csv')
y_test = pd.read_csv('test_target.csv').squeeze()
print("--- 데이터 준비 완료 ---")


# --------------------------------------------------------------------------
# 2. 블렌딩을 위한 데이터 분할 (홀드아웃)
# --------------------------------------------------------------------------
print("\n--- 홀드아웃 방식: 학습 데이터를 80/20으로 분할 ---")
# 전체 학습 데이터를 베이스 모델 학습용(base_train)과 메타 모델 학습용(holdout)으로 분할
X_base_train, X_holdout, y_base_train, y_holdout = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# PyCaret용 DataFrame 생성
y_base_train.name = 'target'
base_train_df = pd.concat([X_base_train, y_base_train], axis=1)


# --------------------------------------------------------------------------
# 3. 베이스 모델 튜닝 및 학습 (base_train 데이터만 사용)
# --------------------------------------------------------------------------
print("\n--- PyCaret 환경 설정 (베이스 모델용) ---")
# 베이스 모델 학습용 데이터로만 PyCaret 환경 설정
clf_setup = setup(data=base_train_df,
                  target='target',
                  session_id=42,
                  verbose=False)

print("\n--- 베이스 모델 하이퍼파라미터 튜닝 시작 ---")
tuned_lgbm = tune_model(create_model('lightgbm', verbose=False), optimize='AUC', verbose=False)
tuned_rf = tune_model(create_model('rf', verbose=False), optimize='AUC', verbose=False)
print("--- 베이스 모델 튜닝 완료 ---")

# 전체 base_train 데이터로 최종 베이스 모델 학습
final_lgbm = finalize_model(tuned_lgbm)
final_rf = finalize_model(tuned_rf)


# --------------------------------------------------------------------------
# 4. 메타 모델 학습 (holdout 데이터만 사용)
# --------------------------------------------------------------------------
print("\n--- 메타 모델 학습 시작 ---")
# holdout 데이터에 대한 베이스 모델들의 예측값을 메타 모델의 학습 데이터로 사용
holdout_preds_lgbm = predict_model(final_lgbm, data=X_holdout)['prediction_score']
holdout_preds_rf = predict_model(final_rf, data=X_holdout)['prediction_score']
meta_X_train = np.c_[holdout_preds_lgbm, holdout_preds_rf]

# 메타 모델 학습
meta_model = LogisticRegression()
meta_model.fit(meta_X_train, y_holdout)
print("--- 메타 모델 학습 완료 ---")


# --------------------------------------------------------------------------
# 5. 최종 예측, 평가 및 저장
# --------------------------------------------------------------------------
print("\n--- 최종 예측 및 평가 ---")
# 1. 테스트 데이터에 대한 베이스 모델 예측
test_preds_lgbm = predict_model(final_lgbm, data=X_test)['prediction_score']
test_preds_rf = predict_model(final_rf, data=X_test)['prediction_score']
meta_X_test = np.c_[test_preds_lgbm, test_preds_rf]

# 2. 메타 모델로 최종 예측
blend_proba = meta_model.predict_proba(meta_X_test)[:, 1]
blend_preds = (blend_proba > 0.5).astype(int)

# 3. 최종 평가 결과 저장 (CSV 파일)
print("\n--- 최종 평가 결과를 CSV 파일에 저장합니다 ---")
report_dict = classification_report(y_test, blend_preds, digits=4, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

roc_auc = roc_auc_score(y_test, blend_proba)
auprc = average_precision_score(y_test, blend_proba)
report_df['roc_auc'] = roc_auc
report_df['auprc'] = auprc

report_df.to_csv("blending_model_results.csv")
print("평가 결과가 blending_model_results.csv 파일에 저장되었습니다.")