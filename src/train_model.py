from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(df):
    features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Fwd Packet Length Max',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow Packets/s'
    ]
    target = 'Label'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    score = accuracy_score(y_test, clf.predict(X_test))
    return clf, score, features, X_test, y_test