import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 데이터가 저장된 디렉토리
base_dir = "C:/Users/KYH/Desktop/deeplearning/RF"

# 파일 경로 리스트
file_paths = [
    "1_data(total)_dh.csv", "2_data(total)_dh.csv", "3_data(total)_dh.csv",  # 보행 데이터 파일 경로
    "4_data(total)_dh.csv", "5_data(total)_dh.csv",
    "1_data(total)_jae.csv", "2_data(total)_jae.csv", "3_data(total)_jae.csv",
    "4_data(total)_jae.csv", "5_data(total)_jae.csv",
    "1_data(total)_jung.csv", "2_data(total)_jung.csv", "3_data(total)_jung.csv",
    "4_data(total)_jung.csv", "5_data(total)_jung.csv",
    "up1_RMS_total(dh).csv", "up2_RMS_total(dh).csv", "up3_RMS_total(dh).csv",  # 계단 오르기 데이터 파일 경로
    "up4_RMS_total(dh).csv", "up5_RMS_total(dh).csv", "up6_RMS_total(dh).csv",
    "up7_RMS_total(dh).csv", "up8_RMS_total(dh).csv", "up8_RMS_total(dh).csv", "up10_RMS_total(dh).csv",
    "up1_RMS_total(g).csv", "up2_RMS_total(g).csv", "up3_RMS_total(g).csv",
    "up4_RMS_total(g).csv", "up5_RMS_total(g).csv", "up6_RMS_total(g).csv",
    "up7_RMS_total(g).csv", "up8_RMS_total(g).csv", "up8_RMS_total(g).csv", "up10_RMS_total(g).csv",
    "up1_RMS_total(s).csv", "up2_RMS_total(s).csv", "up3_RMS_total(s).csv",
    "up4_RMS_total(s).csv", "up5_RMS_total(s).csv", "up6_RMS_total(s).csv",
    "up7_RMS_total(s).csv", "up8_RMS_total(s).csv", "up8_RMS_total(s).csv", "up10_RMS_total(s).csv",
    "down1_RMS_total(dh).csv", "down2_RMS_total(dh).csv", "down3_RMS_total(dh).csv",  # 계단 내리기 데이터 파일 경로
    "down4_RMS_total(dh).csv", "down5_RMS_total(dh).csv", "down6_RMS_total(dh).csv",
    "down7_RMS_total(dh).csv", "down8_RMS_total(dh).csv", "down9_RMS_total(dh).csv", "down10_RMS_total(dh).csv",
    "down1_RMS_total(g).csv", "down2_RMS_total(g).csv", "down3_RMS_total(g).csv",
    "down4_RMS_total(g).csv", "down5_RMS_total(g).csv", "down6_RMS_total(g).csv",
    "down7_RMS_total(g).csv", "down8_RMS_total(g).csv", "down9_RMS_total(g).csv", "down10_RMS_total(g).csv",
    "down1_RMS_total(s).csv", "down2_RMS_total(s).csv", "down3_RMS_total(s).csv",
    "down4_RMS_total(s).csv", "down5_RMS_total(s).csv", "down6_RMS_total(s).csv",
    "down7_RMS_total(s).csv", "down8_RMS_total(s).csv", "down9_RMS_total(s).csv", "down10_RMS_total(s).csv"
]

# 데이터 전처리 함수
def preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = ['Ch 1', 'Ch 2', 'Ch 3', 'Ch 4', 'Ch 5', 'Ch 6']
        if all(col in df.columns for col in required_columns):
            data = df[required_columns].values
            return data
        else:
            return None
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {file_path}, 오류: {e}")
        return None

# 데이터 로드 및 확인
X_list, y_list = [], []
print("데이터 로드 중...")
for i, file_path in enumerate(tqdm(file_paths, desc="파일 로드 진행")):
    full_path = os.path.join(base_dir, file_path)
    if os.path.exists(full_path):
        data = preprocess_data(full_path)
        if data is not None:
            X_list.append(data)
            y_list.append(np.full(data.shape[0], i))  # 0: Walking, 1: Stair Ascent, 2: Stair Descent

# 데이터가 적절히 로드되었는지 확인
if X_list and y_list:
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    print(f"전체 데이터 크기: X = {X.shape}, y = {y.shape}")
else:
    print("모델을 훈련하고 테스트할 유효한 데이터가 없습니다.")
    exit()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 최적 하이퍼파라미터
best_params = {
    'n_estimators': 300,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}

# 교차검증
print("playing cross validation please wait...")
final_model = RandomForestClassifier(**best_params, random_state=42)

# 교차검증 점수 계산 (정확도와 F1 스코어 모두 평가)
cv_results = cross_validate(
    final_model,
    X_train,
    y_train,
    cv=5,  # 5-fold 교차검증
    scoring=['accuracy', 'f1_weighted'],  # 정확도와 F1 스코어 계산
    return_train_score=True
)

# 교차검증 결과 출력
print("\ncross validation result:")
print(f"  average train_accuracy: {np.mean(cv_results['train_accuracy']):.4f}")
print(f"  average test_accuray: {np.mean(cv_results['test_accuracy']):.4f}")
print(f"  average F1 score(weight): {np.mean(cv_results['test_f1_weighted']):.4f}")

# 모델 학습 및 테스트 데이터 평가
#최종 모델 훈련 및 테스트 데이터 평가...
print("\nfinal model, test data validation")
final_model.fit(X_train, y_train)
y_test_pred = final_model.predict(X_test)

# 테스트 데이터 성능 평가
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1_score = f1_score(y_test, y_test_pred, average=None)

print("\nTest data performance:")
print(f"  prdict: {test_accuracy:.4f}")
print("  F1 score (tpye class):")
for i, score in enumerate(test_f1_score):
    print(f"    class {i}: {score:.4f}")

# 혼동 행렬 시각화
print("\nVisualizing confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_test_pred, labels=np.unique(y_test))

# 일반 혼동 행렬
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Walking', 'Stair Ascent', 'Stair Descent'],
            yticklabels=['Walking', 'Stair Ascent', 'Stair Descent'])
plt.xlabel("predict label")
plt.ylabel("real label")
plt.title("Confusion matrix (Raw)")
plt.show()

# 정규화된 혼동 행렬
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=['Walking', 'Stair Ascent', 'Stair Descent'],
            yticklabels=['Walking', 'Stair Ascent', 'Stair Descent'])
plt.xlabel("predict label")
plt.ylabel("real label")
plt.title("Normalized Confusion Matrix")
plt.show()
