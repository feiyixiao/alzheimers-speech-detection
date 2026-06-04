from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_models():
    return {
        "LogReg":  LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
        "SVM":     SVC(kernel="rbf", probability=True, random_state=42),
        "RF":      RandomForestClassifier(n_estimators=100, random_state=42),
        "MLP":     MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }
