import os
import random

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, recall_score, roc_auc_score, \
    average_precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.utils import class_weight

from StackIRESFinding.models import StackIRESFinding

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
# 预读取数据
train = np.load('../dataset/train/train_set.npz')
X_tra, y_tra = train['X_tra'], train['y_tra']

name = 'new_StackWeight'
# 计算权重
all_cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_tra), y=y_tra.reshape(-1))
cw = dict(enumerate(all_cw))
# 设定记录指标的数组
acc = np.zeros(5)
sn = np.zeros(5)
sp = np.zeros(5)
mcc = np.zeros(5)
f1 = np.zeros(5)
auroc = np.zeros(5)
aupr = np.zeros(5)
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []
model_prediction_result_cv = []
kfold = KFold(n_splits=5, shuffle=True,
              # 试试种子不固定
              random_state=42)

for i, (tra, val) in enumerate(kfold.split(X_tra, y_tra)):
    # 每一折数据划分
    X_train = X_tra[tra]
    y_train = y_tra[tra]
    X_val = X_tra[val]
    y_val = y_tra[val]
    model = StackIRESFinding()
    weightPath = 'weights/5cv/' + name + f'/{i + 1}/'
    checkpoint = ModelCheckpoint(weightPath, verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='max', monitor='val_auc')
    earlyStop = EarlyStopping(monitor='val_auc', patience=20, verbose=1, mode='max')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, class_weight=cw,
              callbacks=[checkpoint, earlyStop])

    model.load_weights(weightPath).expect_partial()

    # 计算结果
    print(f'第{i + 1}折的结果')

    y_pred1 = model.predict(X_val)
    y_pred = np.where(y_pred1 > 0.5, 1, 0)
    y_pred = y_pred.reshape(-1, )
    y_pred1 = y_pred1.reshape(-1, )
    y_val = y_val.reshape(-1, )
    tmp_result = np.zeros((len(y_val), 3))
    tmp_result[:, 0], tmp_result[:, 1], tmp_result[:, 2] = y_val, y_pred, y_pred1
    model_prediction_result_cv.append(tmp_result)
    acc[i] = accuracy_score(y_val, y_pred)
    sn[i] = recall_score(y_val, y_pred)
    mcc[i] = matthews_corrcoef(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    sp[i] = tn / (tn + fp)
    auroc[i] = roc_auc_score(y_val, y_pred1)
    aupr[i] = average_precision_score(y_val, y_pred1)
    f1[i] = f1_score(y_val, np.round(y_pred1.reshape(-1)))
    fpr, tpr, threshold = roc_curve(y_val, y_pred1)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print("ACC : ", acc[i])
    print("SN : ", sn[i])
    print("SP : ", sp[i])
    print("MCC : ", mcc[i])
    print("AUC : ", auroc[i])
    print("AUPR : ", aupr[i])
    print("f1_score : ", f1[i])
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
std_auc = np.std(tprs, axis=0)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()
print("均值")
print("ACC : ", np.mean(acc))
print("SN : ", np.mean(sn))
print("SP : ", np.mean(sp))
print("MCC : ", np.mean(mcc))
print("AUC : ", np.mean(auroc))
print("AUPR : ", np.mean(aupr))
print("f1_score : ", np.mean(f1))
