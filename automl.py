from flaml import AutoML
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# 1) 데이터 로드 (PMLB adult: 이진분류)
df = fetch_data('adult', return_X_y=False)  # target 컬럼 포함된 DataFrame
# PMLB adult는 수치형으로 정리돼 있음 (환경에 따라 비수치가 섞이면 one-hot 해도 됨)
y = df['target']
X = df.drop(columns=['target'])

# 혹시 비수치 컬럼이 있다면 주석 해제해서 원-핫 인코딩
# if any(dtype == 'object' for dtype in X.dtypes):
#     X = pd.get_dummies(X, drop_first=True)

# 2) 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) AutoML 학습 (시간 예: 60초, 정확도 기준)
automl = AutoML()
automl.fit(
    X_train=X_train, y_train=y_train,
    task="classification",
    time_budget=60,            # 초 단위
    metric="accuracy",         # 필요시 'f1', 'roc_auc' 등으로 변경
    log_file_name="flaml.log", # 로그 파일
    # 알고리즘 제한하고 싶으면 다음 라인 추가:
    # estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "lrl1"]
)

# 4) 평가
y_pred = automl.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 5) 베스트 모델/설정 확인
print("Best estimator:", automl.best_estimator)
print("Best config:", automl.best_config)

# 기존: automl.best_validation_score  -> 존재하지 않음
# 대체 1) loss 그대로 보기
print("Best validation loss:", automl.best_loss)

# 대체 2) metric='accuracy'를 썼으니 검증 정확도로 환산
print(f"Best validation accuracy: {1 - automl.best_loss:.4f}")

# 참고: 자세한 결과 딕셔너리
print("Best result dict:", automl.best_result)  # 훈련/검증 관련 요약 제공
