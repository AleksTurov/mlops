import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from catboost import Pool


def calibrate_model(model, X_train, y_train, method):
    """
    Калибрует модель с использованием указанных методов.
    
    Возвращает калиброванную модель и предсказанные вероятности.
    """
    calibrated_model = CalibratedClassifierCV(estimator=model, method=method, cv='prefit')
    calibrated_model.fit(X_train, y_train)
    probs_calibrated = calibrated_model.predict_proba(X_train)[:, 1]
    return calibrated_model, probs_calibrated

def plot_calibration_curve(y_true, clf_list):
    """
    Строит графики калибровки для различных методов калибровки.
    """
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for probs, name in clf_list:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=10)
        brier = brier_score_loss(y_true, probs)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{name} (Brier: {brier:.3f})")

    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.legend(loc="lower right")
    plt.title('Calibration plots (reliability curve)')
    plt.show()