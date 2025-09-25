import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path="Final Transactions.csv"):
    # 데이터 불러오기
    df = pd.read_csv(file_path)

    # 사기거래/합법거래 양 비교
    print(df['TX_FRAUD'].value_counts())
    print(df['TX_FRAUD'].value_counts(normalize=True))

    # --- 특징 공학 ---
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
    df['TX_DAY_OF_WEEK'] = df['TX_DATETIME'].dt.dayofweek

    # --- 불필요한 컬럼 제거 ---
    # Unnamed: 0: 인덱스 컬럼으로 불필요
    # TRANSACTION_ID, CUSTOMER_ID, TERMINAL_ID: 단순 식별자로 모델 성능에 영향 X
    # TX_DATETIME: 숫자형 특징(시간, 요일)으로 대체했으므로 제거
    # TX_FRAUD_SCENARIO: 사기 유형 정보로, 타겟 변수(TX_FRAUD)에 대한 정답을 알려주는 컬럼이므로 반드시 제거
    df = df.drop(columns=[
        'Unnamed: 0',
        'TRANSACTION_ID',
        'TX_DATETIME',
        'CUSTOMER_ID',
        'TERMINAL_ID',
        'TX_FRAUD_SCENARIO'
    ])

    # --- 데이터 분할 ---
    X = df.drop('TX_FRAUD', axis=1)
    y = df['TX_FRAUD']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"학습 데이터 형태: {X_train.shape}")
    print(f"테스트 데이터 형태: {X_test.shape}")

    return X_train, X_test, y_train, y_test
