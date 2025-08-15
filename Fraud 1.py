import pandas as pd
data = pd.read_csv("creditcard.csv")
print(data.head())
print("Shape:", data.shape)
print(data["Class"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate input features (X) and target column (y)
X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data:", X_train.shape, " Testing data:", X_test.shape)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

print("Counts after balancing:", y_train.value_counts())

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

logit = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None)
logit.fit(X_train, y_train)
y_pred_logit = logit.predict(X_test)
print(classification_report(y_test, y_pred_logit, digits=4))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_logit)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


rf = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest Report ---")
print(classification_report(y_test, y_pred_rf, digits=4))


xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric="logloss"
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n--- XGBoost Report ---")
print(classification_report(y_test, y_pred_xgb, digits=4))

from sklearn.metrics import roc_curve, auc, precision_recall_curve


y_prob_logit = logit.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_logit)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


precision, recall, _ = precision_recall_curve(y_test, y_prob_logit)
plt.figure()
plt.plot(recall, precision, label="Logistic Regression")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()


import joblib


joblib.dump(rf, "fraud_rf_model.pkl")





