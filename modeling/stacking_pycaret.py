import pandas as pd
from pycaret.classification import *
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------
# 1. 데이터 준비: 전처리된 데이터를 CSV 파일에서 불러옵니다.python stacking_pycaret.py
# (이전에 만든 preprocessing.py를 먼저 실행해서 파일을 만들어 두어야 합니다.)
# --------------------------------------------------------------------------
print("--- 전처리된 데이터 불러오기 시작 ---")
X_train = pd.read_csv('train_features.csv')
y_train = pd.read_csv('train_target.csv').squeeze() # squeeze()로 Series 변환
X_test = pd.read_csv('test_features.csv')
y_test = pd.read_csv('test_target.csv').squeeze()

# PyCaret의 setup 함수는 피처와 타겟이 합쳐진 DataFrame을 필요로 합니다.
y_train.name = 'target' # 타겟 Series의 이름을 지정해야 합니다.
train_df = pd.concat([X_train, y_train], axis=1)
print("--- 데이터 준비 완료 ---")


# --------------------------------------------------------------------------
# 2. PyCaret 환경 설정
# --------------------------------------------------------------------------
print("\n--- PyCaret 환경 설정 시작 ---")
# 데이터를 분석하고 전처리 파이프라인을 설정합니다.
# GPU가 있다면 use_gpu=True 옵션을 추가할 수 있습니다.
clf_setup = setup(data=train_df, 
                  target='target', 
                  session_id=42, # 결과 재현을 위한 시드
                  verbose=False) # 과정 출력 최소화
print("--- PyCaret 환경 설정 완료 ---")


# --------------------------------------------------------------------------
# 3. 베이스 모델 생성 및 하이퍼파라미터 튜닝
# --------------------------------------------------------------------------
print("\n--- 베이스 모델 하이퍼파라미터 튜닝 시작 ---")
# create_model로 모델을 불러온 후, tune_model로 하이퍼파라미터를 최적화합니다.
# 평가지표 'AUC'를 기준으로 최적의 파라미터를 찾습니다.
tuned_lgbm = tune_model(create_model('lightgbm', verbose=False), optimize='AUC', verbose=False)
tuned_rf = tune_model(create_model('rf', verbose=False), optimize='AUC', verbose=False)
print("--- 베이스 모델 튜닝 완료 ---")


# --------------------------------------------------------------------------
# --- 4. 스태킹(블렌딩) 모델 생성 ---
print("\n--- 스태킹(블렌딩) 모델 생성 시작 ---")

# 2. 메타 모델로 사용할 실제 객체를 생성
final_estimator = LogisticRegression() 

# 3. meta_model 인자에 문자열 'lr' 대신 생성한 객체를 전달
stacker = stack_models(estimator_list=[tuned_lgbm, tuned_rf], 
                         meta_model=final_estimator,
                         verbose=False)
                         
print("--- 스태킹(블렌딩) 모델 생성 완료 ---")


# --------------------------------------------------------------------------
# 5. 최종 모델 학습 및 평가
# --------------------------------------------------------------------------
print("\n--- 최종 모델 학습 및 평가 ---")
# 전체 학습 데이터로 최종 모델을 다시 학습합니다.
final_stack_model = finalize_model(stacker)

# 테스트 데이터(X_test)로 예측을 수행합니다.
predictions = predict_model(final_stack_model, data=X_test)

# 평가
# PyCaret 3.x 버전 기준 컬럼명: 'prediction_label' (예측값), 'prediction_score' (Positive 확률)
y_test_pred = predictions['prediction_label']
y_test_proba = predictions['prediction_score']


# --- 최종 평가 결과 저장 (CSV 파일) ---
print("\n--- 최종 평가 결과를 CSV 파일에 저장합니다 ---")

# 1. classification_report를 딕셔너리 형태로 받음
report_dict = classification_report(y_test, y_test_pred, digits=4, output_dict=True)

# 2. 딕셔너리를 DataFrame으로 변환
report_df = pd.DataFrame(report_dict).transpose()

# 3. 추가 지표들도 DataFrame에 추가
roc_auc = roc_auc_score(y_test, y_test_proba)
auprc = average_precision_score(y_test, y_test_proba)
report_df['roc_auc'] = roc_auc
report_df['auprc'] = auprc

# 4. CSV 파일로 저장
report_df.to_csv("stacking_model_results.csv")
print("평가 결과가 stacking_model_results.csv 파일에 저장되었습니다.")

# --- 학습된 최종 모델 저장 ---
save_model(final_stack_model, 'my_final_stacking_model')
print("학습된 모델이 my_final_blending_model.pkl 파일로 저장되었습니다.")