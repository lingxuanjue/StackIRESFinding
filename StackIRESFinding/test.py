import numpy as np
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, confusion_matrix, roc_auc_score, \
    f1_score

from StackIRESFinding.models import StackIRESFinding

test_balance = np.load('../dataset/test/balanced_test_set.npz')
test_imbalance = np.load('../dataset/test/imbalanced_test_set.npz')

X_tes_balance, y_tes_balance = test_balance['X_tes'], test_balance['y_tes']
X_tes_imbalance, y_tes_imbalance = test_imbalance['X_tes'], test_imbalance['y_tes']

model = StackIRESFinding()
name = 'StackIRESFinding'

prob_balance = []
prob_imbalance = []
# 平衡测试集
for i in range(0, 5):
    weightPath = 'weights/5cv/' + name + f'/{i + 1}/'
    model.load_weights(weightPath).expect_partial()
    y_prob_balance = model.predict(X_tes_balance)[:, 0]  # 获取概率预测
    prob_balance.append(y_prob_balance)

prob_balance = np.array(prob_balance)
avg_prob_balance = np.mean(prob_balance, axis=0)  # 对10个模型的概率取平均
final_pred_balance = np.where(avg_prob_balance > 0.5, 1, 0)  # 根据平均概率进行最终预测

# 计算各项指标
acc = accuracy_score(y_tes_balance, final_pred_balance)
sn = recall_score(y_tes_balance, final_pred_balance)
mcc = matthews_corrcoef(y_tes_balance, final_pred_balance)
tn, fp, fn, tp = confusion_matrix(y_tes_balance, final_pred_balance).ravel()
sp = tn / (tn + fp)
f1 = f1_score(y_tes_balance, final_pred_balance.reshape(-1))
auc_score = roc_auc_score(y_tes_balance, avg_prob_balance)  # 计算AUC

print("=================平衡软投票结果=================")
print("ACC : ", acc)
print("SN : ", sn)
print("SP : ", sp)
print("MCC : ", mcc)
print("F1-sorce : ", f1)
print("AUC : ", auc_score)

# 不平衡测试集
for i in range(0, 5):
    weightPath = 'weights/5cv/' + name + f'/{i + 1}/'
    model.load_weights(weightPath).expect_partial()
    y_prob_imbalance = model.predict(X_tes_imbalance)[:, 0]  # 获取概率预测
    prob_imbalance.append(y_prob_imbalance)

prob_imbalance = np.array(prob_imbalance)
avg_prob_imbalance = np.mean(prob_imbalance, axis=0)  # 对10个模型的概率取平均
final_pred_imbalance = np.where(avg_prob_imbalance > 0.5, 1, 0)  # 根据平均概率进行最终预测

# 计算各项指标
acc = accuracy_score(y_tes_imbalance, final_pred_imbalance)
sn = recall_score(y_tes_imbalance, final_pred_imbalance)
mcc = matthews_corrcoef(y_tes_imbalance, final_pred_imbalance)
tn, fp, fn, tp = confusion_matrix(y_tes_imbalance, final_pred_imbalance).ravel()
sp = tn / (tn + fp)
f1 = f1_score(y_tes_imbalance, final_pred_imbalance.reshape(-1))
auc_score = roc_auc_score(y_tes_imbalance, avg_prob_imbalance)  # 计算AUC

print("=================不平衡软投票结果=================")
print("ACC : ", acc)
print("SN : ", sn)
print("SP : ", sp)
print("MCC : ", mcc)
print("F1-sorce : ", f1)
print("AUC : ", auc_score)
