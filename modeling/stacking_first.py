from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# 전처리 함수 불러오기
from preprocessing import load_and_preprocess_data

# 데이터 로드
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# --- 스태킹 모델 정의 ---
estimators = [
    ('lgbm', LGBMClassifier(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
]
final_estimator = LogisticRegression()

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1
)

# --- 모델 학습 및 평가 ---
print("\n--- 스태킹 모델 학습 및 평가 (ROC-AUC Score, AUPRC) ---")
stack_model.fit(X_train, y_train)

stack_preds = stack_model.predict(X_test)
stack_proba = stack_model.predict_proba(X_test)[:, 1]

print("스태킹 모델 평가 결과:")
print(classification_report(y_test, stack_preds, digits=4))
print(f"ROC-AUC Score: {roc_auc_score(y_test, stack_proba):.4f}")
print(f"AUPRC: {average_precision_score(y_test, stack_proba):.4f}")
