import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("../data/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, pred)

print("ROC AUC Score:", auc)
