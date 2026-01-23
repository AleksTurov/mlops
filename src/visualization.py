from catboost.utils import get_roc_curve, get_fpr_curve, get_fnr_curve
import matplotlib.pyplot as plt
import numpy as np
from src.logger import logger
import pandas as pd
import seaborn as sns
import os
from sklearn.calibration import calibration_curve
import mlflow

from catboost import Pool
import shap
from shap import TreeExplainer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_score,
    average_precision_score,
    recall_score,
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay
)
from src.modeling import objective
from operator import itemgetter



def fpr_fnr(y_train, probs, filename='fpr_fnr_plot.png', 
            save_flg=True, 
            plt_show=False, artifacts_dir=f'./docs/', 
            ):
    """
    Функция для построения кривых FPR и FNR в зависимости от порога классификации.
    Arguments:
    y_train -- истинные метки классов для обучающего набора.
    probs -- предсказанные вероятности положительного класса.
    filename -- имя файла для сохранения графика.
    save_flg -- флаг для сохранения графика.
    plt_show -- флаг для отображения графика.
    artifacts_dir -- директория для сохранения графика.   
    """
    # Расчет ROC-кривой
    fpr, tpr, thresholds = roc_curve(y_train, probs)
    
    # Расчет FNR
    fnr = 1 - tpr
    
    plt.figure(figsize=(16, 8))
    lw = 2
    plt.plot(thresholds, fpr, color='blue', lw=lw, label='FPR', alpha=0.5)
    plt.plot(thresholds, fnr, color='green', lw=lw, label='FNR', alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.xlabel('Thresholds', fontsize=16)
    plt.ylabel('Error rate', fontsize=16)
    plt.title('FPR-FNR curves', fontsize=16)
    plt.legend(loc='lower left', fontsize=16)
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, filename)
    if save_flg:
        plt.savefig(path, bbox_inches='tight')
    if plt_show:
        # Display the plot
        plt.show()
        plt.close()
    # Найти оптимальный порог, минимизирующий сумму FPR и FNR
    optimal_idx = np.argmin(fpr + fnr)
    optimal_threshold = thresholds[optimal_idx]
    logger.info(f'Optimal threshold: {optimal_threshold}')
    # Найти порог максимизирующий F1 score
    f1_scores = []
    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        f1 = f1_score(y_train, y_pred)
        f1_scores.append(f1)
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds[optimal_f1_idx]
    logger.info(f'Optimal F1 threshold: {optimal_f1_threshold}')
    return optimal_threshold, optimal_f1_threshold, path

# %%
def get_feature_names(model, X_train):
    """
    Function to get feature names from the model or the training data.

    Arguments:
    model -- Trained model.
    X_train -- Training data.

    Returns:
    feature_names -- List of feature names.
    """
    if hasattr(model, 'feature_names_'):
        return model.feature_names_
    else:
        return X_train.columns.tolist()
def plot_confusion_matrix(y, y_proba, optimal_threshold, save_flg=True, show_flg=True, filename="confusion_matrix.png", artifacts_dir='./docs/', title_text='Confusion Matrix'):
    """
    Function to plot the confusion matrix using the optimal threshold.

    Arguments:
    final_model -- trained CatBoostClassifier model.
    X -- input features for predictions.
    y -- true class labels.
    optimal_threshold -- optimal threshold for binary classification.
    categorical_features -- list of categorical features.
    save_flg -- boolean flag indicating whether to save the plot.
    filename -- name of the file to save the plot.
    artifacts_dir -- directory to save the plot.
    
    Returns:
    fig -- the figure object containing the confusion matrix.
    """
    
    # Apply the optimal threshold to get predictions
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Create a ConfusionMatrixDisplay instance
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(title_text, fontsize=6)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, filename)
    if save_flg:
        plt.savefig(path, bbox_inches='tight')
    # only show if interactive requested
    if show_flg:
        plt.show()
        plt.close()
    return fig, path

color_list = ["red", "blue", "green", "orange", "black", "purple"]


def gini(x, y):
    return 2 * auc(x, y) - 1


def plot_roc_curve(y_all, metric="auc", show_flg=True, save_flg=True, filename="roc.jpg", artifacts_dir='./docs/'):
    ''' Функция для построения ROC-кривой и вычисления AUC или GINI.'''

    if metric == "auc":
        metric_label = "AUC"
        metric_function = auc
    elif metric == "gini":
        metric_label = "GINI"
        metric_function = gini

    fig = plt.figure(figsize=(12, 8))
    lw = 2

    for i, y_label in enumerate(y_all):
        y_true, y_pred = np.array(y_all[y_label][0]), np.array(y_all[y_label][1])
        [fpr, tpr, tresholds] = roc_curve(y_true, y_pred)
        metric_value = metric_function(fpr, tpr)
        color = color_list[i % len(color_list)]
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=lw,
            label=f"{metric_label} {y_label} {round(metric_value, 3)}",
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.title("Roc curve", fontsize=18)
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right", fontsize=14)
    path = os.path.join(artifacts_dir, filename)
    if save_flg:
        plt.savefig(path, bbox_inches='tight')
    if show_flg:
        plt.show()
    return fig, path


def plot_precision_recall(y_all, show_flg=True, save_flg=True, filename="precision_recall.jpg", artifacts_dir='./docs/'):

    metric_label = "AP"
    metric_function = average_precision_score

    fig = plt.figure(figsize=(12, 8))
    lw = 2

    for i, y_label in enumerate(y_all):
        y_true, y_pred = np.array(y_all[y_label][0]), np.array(y_all[y_label][1])
        [precision, recall, tresholds] = precision_recall_curve(y_true, y_pred)
        metric_value = metric_function(y_true, y_pred)
        color = color_list[i % len(color_list)]
        plt.plot(
            recall,
            precision,
            color=color,
            lw=lw,
            label=f"{metric_label} {y_label} {round(metric_value, 3)}",
        )

    plt.title("Precision Recall curve", fontsize=18)
    plt.xlabel("Recall", fontsize=18)
    plt.ylabel("Precision", fontsize=18)
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right", fontsize=14)
    path = os.path.join(artifacts_dir, filename)
    if save_flg:
        plt.savefig(path, bbox_inches='tight')
    if show_flg:
        plt.show()
    return fig, path


def cum_gain(y_true, y_prob):
    """Вычисляется кумулятивный gain
    Параметры:
        y_true -- флаги истиных ответов,
        y_prob -- вероятность полученная от модели
    """
    d = pd.DataFrame({"y_true": list(y_true), "y_prob": list(y_prob)})
    d.sort_values(inplace=True, by=["y_prob"], ascending=False)
    d["y_cumsum"] = d.y_true.cumsum()
    d["y_rate"] = np.asarray(d["y_cumsum"]) / sum(d["y_true"])
    d["x_rate"] = (np.arange(len(d)) + 1) / len(d)
    d["y_best_rate"] = np.cumsum(
        [1] * int(sum(d["y_true"])) + [0] * int(len(d) - sum(d["y_true"]))
    ) / sum(d["y_true"])
    d = d.drop_duplicates('y_prob', keep='last')
    return (
        np.append(0, d["x_rate"].values),
        np.append(0, d["y_rate"].values),
        d["y_best_rate"].values,
    )


def plot_cum_gain_curve(y_all, show_flg=True, save_flg=True, filename="cumgain.jpg", artifacts_dir='./docs/'):

    fig = plt.figure(figsize=(12, 8))
    lw = 2

    for i, y_label in enumerate(y_all):
        y_true, y_pred = np.array(y_all[y_label][0]), np.array(y_all[y_label][1])

        supp, tpr, best_tpr = cum_gain(y_true, y_pred)
        plt.plot(supp, tpr, color=color_list[i], lw=lw, label=y_label)

    plt.xlabel("% customers", fontsize=18)
    plt.ylabel("% customers affinity", fontsize=18)
    plt.title("Cumulative Gains Curve", fontsize=18)
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right", fontsize=14)
    path = os.path.join(artifacts_dir, filename)
    if save_flg:
        plt.savefig(path, bbox_inches='tight')
    if show_flg:
        plt.show()
    return fig, path


def lift_metrics(y_true, y_prob):
    """Вычисляется lift
    Параметры:
        y_true -- флаги истиных ответов,
        y_prob -- вероятность полученная от модели
    """
    percentages, gains, best_tpr = cum_gain(y_true, y_prob)
    percentages = percentages[1:]
    gains = gains[1:] / percentages

    return percentages, gains


def plot_lift_curve(y_all, show_flg=False, save_flg=True, filename="lift.jpg", artifacts_dir='./docs/'):

    fig = plt.figure(figsize=(12, 8))
    lw = 2

    for i, y_label in enumerate(y_all):
        y_true, y_pred = np.array(y_all[y_label][0]), np.array(y_all[y_label][1])

        percentages, gains = lift_metrics(y_true, y_pred)
        plt.plot(percentages, gains, color=color_list[i], lw=lw, label=y_label)

    plt.plot([0, 1], [1, 1], "k--", lw=2, label="Baseline")
    plt.plot([0, 1], [3, 3], "k--", lw=2, label="LIFT=3")
    plt.title("Lift curve", fontsize=18)
    plt.xlabel("% customers", fontsize=18)
    plt.ylabel("Lift", fontsize=18)
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend(loc="upper right", fontsize=14)
    path = os.path.join(artifacts_dir, filename)
    if save_flg:
        plt.savefig(path, bbox_inches='tight') 
    if show_flg:
        plt.show()
    return fig, path

def plot_buckets(
    preds_old,
    preds_new,
    kind="buckets",
    n_bins=10,
    label="",
    scale="rate",
    label_old="Старый",
    label_new="Новый",
    show_flg=True,
    save_flg=True,
    filename='buckets.jpg',
    ndigits=2,
    artifacts_dir = './docs/'
):
    """
    kind: {'buckets', 'parity'}
    scale: {'rate', 'perc'}
    """

    preds_old = np.array(preds_old)
    preds_new = np.array(preds_new)

    if scale == "rate":
        scale_label = ""
        scale_coef = 1
        ndigits = 4
    elif scale == "perc":
        scale_label = "%"
        scale_coef = 100

    if kind == "buckets":
        sorted_old = np.array(sorted(preds_old, reverse=True))
        sorted_new = np.array(sorted(preds_new, reverse=True))
    elif kind == "parity":
        sorted_ind = np.array(
            sorted(list(enumerate(preds_new)), key=itemgetter(1), reverse=True)
        )[:, 0].astype("int")
        sorted_old = preds_old[sorted_ind]
        sorted_new = preds_new[sorted_ind]

    splitted_old = np.array_split(sorted_old, n_bins)
    splitted_new = np.array_split(sorted_new, n_bins)

    mean_bin_old = [scale_coef * np.mean(bin) for bin in splitted_old]
    mean_bin_new = [scale_coef * np.mean(bin) for bin in splitted_new]

    fig, ax = plt.subplots(figsize=(16, 8))

    width = 0.44
    rects1 = ax.bar(
        np.array(range(1, len(mean_bin_old) + 1)) - width / 2,
        mean_bin_old,
        width=width,
        label=label_old,
    )
    rects2 = ax.bar(
        np.array(range(1, len(mean_bin_new) + 1)) + width / 2,
        mean_bin_new,
        width=width,
        label=label_new,
    )

    for rect in rects1:
        h = rect.get_height()
        ax.text(
            rect.get_x(),
            1.02 * h,
            str(f"{h:.{ndigits}f}" + scale_label),
            va="bottom",
            fontsize=9,
        )
    for rect in rects2:
        h = rect.get_height()
        ax.text(
            rect.get_x(),
            1.02 * h,
            str(f"{h:.{ndigits}f}" + scale_label),
            va="bottom",
            fontsize=9,
        )

    ax.legend(fontsize=12)
    ax.set_xlabel("Бакет", fontsize=14)
    ax.set_xticks(range(1, 11))
    ax.set_title(label, fontsize=20)
    filename = f"{kind}.jpg"
    path = os.path.join(artifacts_dir, filename)
    if save_flg:
        plt.savefig(path, bbox_inches='tight')
    if show_flg:
        plt.show()
    return fig, path



def compute_shap_values(final_model, X_train, y_train, categorical_features, class_idx=1):
    """
    Function to compute SHAP values for a CatBoost model.

    Arguments:
    final_model -- Trained CatBoostClassifier model.
    X_train -- Training data.
    y_train -- Class labels for training data.
    categorical_features -- List of categorical features.
    class_idx -- Index of the class for multiclass models (default is 1).

    Returns:
    shap_values_class -- SHAP values for the specified class.
    base_value -- Base value for SHAP values.
    """
    # Create a Pool object for CatBoost data
    train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
    
    # Initialize TreeExplainer for CatBoost
    explainer = shap.TreeExplainer(final_model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(train_pool)
    
    # Determine if the model is binary or multiclass
    if isinstance(shap_values, list):
        # Multiclass case: shap_values is a list of arrays (one for each class)
        shap_values_class = shap_values[class_idx]
        base_value = explainer.expected_value[class_idx]
    else:
        # Binary case: shap_values is a single array
        shap_values_class = shap_values
        base_value = explainer.expected_value
    
    return shap_values_class, base_value


def visualize_shap_values(shap_values, base_value, X_train, artifacts_dir='./', display_plots=False, save_flag=False):
    """
    Function to visualize SHAP values using both a waterfall plot and a summary plot.

    Arguments:
    shap_values -- SHAP values for the model.
    base_value -- Base value for the SHAP values.
    X_train -- Training data used for visualization.
    artifacts_dir -- Directory to save artifacts like plots and other files (default is './').
    display_plots -- Flag to control whether to display plots (default is True).
    
    Returns:
    saved_files -- A list of file paths where the plots were saved.
    """
    # Initialize JavaScript for SHAP in Jupyter (necessary for interactive plots in notebooks)
    shap.initjs()

    # Ensure artifacts directory exists
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)
    
    saved_files = []

    # Waterfall plot visualization for the first and last samples
    for sample_idx in [0, X_train.shape[0] - 1]:
        plt.figure()  # Create a new figure for each plot
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[sample_idx], 
                             base_values=base_value, 
                             data=X_train.iloc[sample_idx]),
            show=display_plots  # Ensure plot is not displayed before saving
        )
        plt.tight_layout()  # Adjust layout to fit elements properly
        # Save the plot
        if save_flag:
            waterfall_plot_path = os.path.join(artifacts_dir, f"shap_waterfall_plot_{sample_idx}.jpg")
            plt.savefig(waterfall_plot_path, format='jpeg', bbox_inches='tight')
            logger.info(f"Waterfall plot for sample index {sample_idx} saved to: {waterfall_plot_path}")
            saved_files.append(waterfall_plot_path)

        plt.close()  # Explicitly close the plot to free up memory

    # Summary plot visualization for SHAP values
    plt.figure()  # Create a new figure for the summary plot
    shap.summary_plot(shap_values, X_train, show=display_plots)  # Set show to False to avoid displaying before saving
    plt.tight_layout()  # Adjust layout
    # Save the summary plot
    if save_flag:
        summary_plot_path = os.path.join(artifacts_dir, "shap_summary_plot.jpg")
        plt.savefig(summary_plot_path, format='jpeg', bbox_inches='tight')
        logger.info(f"Summary plot saved to: {summary_plot_path}")
        saved_files.append(summary_plot_path)

    plt.close()  # Explicitly close the plot to free up memory

    return saved_files   # Return the list of saved files


def plot_calibration_curve(y_true, prob_sets, filename='calibration_curve.png', 
                           artifacts_dir='./docs', n_bins=20, save_flag=True,
                           plt_show=False):
    """
    Построение калибровочных кривых.
    """
    plt.figure(figsize=(10, 8))
    
    for probs, method in prob_sets:
        frac_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=n_bins)
        plt.plot(mean_predicted_value, frac_of_positives, "s-", label=method)
    
    plt.plot([0, 1], [0, 1], "k:", label="Идеально откалибровано")
    plt.ylabel("Доля положительных")
    plt.xlabel("Среднее предсказанное значение")
    plt.title("Калибровочные кривые")
    plt.legend()
    plt.grid()
    plt.show()
    path = os.path.join(artifacts_dir, filename)
    if save_flag:
        plt.savefig(path, bbox_inches='tight')
    if plt_show:
        
        plt.show()
        plt.close()
    return path

def get_save_metrics(X_train, y_train, train_proba, y_oot, oot_proba, 
                    BETA, best_params, optimal_threshold, save_to_mlflow=False):
    ''' 
    Функция для получения и сохранения метрик модели.
    '''    
    results = pd.DataFrame()               # гарантированно существует при возврате

    
    try:
        # Получение предсказаний вероятностей
        train_pred_class = (train_proba >= optimal_threshold).astype(int)
        # Make predictions using the trained model
        oot_pred_class = (oot_proba >= optimal_threshold).astype(int)
        # Calculate metrics
        train_metric = roc_auc_score(y_train, train_proba)
        precision = precision_score(y_train, train_pred_class, zero_division=0)
        recall = recall_score(y_train, train_pred_class, zero_division=0)
        f1 = f1_score(y_train, train_pred_class, zero_division=0)
        accuracy = accuracy_score(y_train, train_pred_class)
            
        # Calculate metrics for oot
        oot_metric = roc_auc_score(y_oot, oot_proba)
        precision_oot = precision_score(y_oot, oot_pred_class, zero_division=0)
        recall_oot = recall_score(y_oot, oot_pred_class, zero_division=0)
        f1_oot = f1_score(y_oot, oot_pred_class, zero_division=0)
        accuracy_oot = accuracy_score(y_oot, oot_pred_class)
        # Calculate objective_metric
        objective_metric = -objective(train_metric, oot_metric, BETA)
        results = pd.DataFrame([[
                len(X_train.columns), 
                train_metric, 
                precision, 
                recall, 
                f1, 
                accuracy, 
                oot_metric, 
                precision_oot, 
                recall_oot, 
                f1_oot, 
                accuracy_oot, 
                objective_metric
            ]], columns=[
                'Number of columns', 
                'ROC AUC TRAIN', 
                'Precision TRAIN', 
                'Recall TRAIN', 
                'F1-Score TRAIN', 
                'Accuracy TRAIN', 
                'ROC AUC OOT', 
                'Precision OOT', 
                'Recall OOT', 
                'F1-Score OOT', 
                'Accuracy OOT', 
                'objective'
            ])
        logger.info("Metrics computed successfully.")
        if save_to_mlflow:
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
                logger.info(f"Logged parameter: {param_name} = {param_value}")
            # Log artifacts and metrics to MLflow
            for index, row in results.iterrows():
                mlflow.log_metric("number_of_columns", row['Number of columns'])
                mlflow.log_metric("roc_auc_train", row['ROC AUC TRAIN'])
                mlflow.log_metric("precision_train", row['Precision TRAIN'])
                mlflow.log_metric("recall_train", row['Recall TRAIN'])
                mlflow.log_metric("f1_score_train", row['F1-Score TRAIN'])
                mlflow.log_metric("accuracy_train", row['Accuracy TRAIN'])
                
                mlflow.log_metric("roc_auc_oot", row['ROC AUC OOT'])
                mlflow.log_metric("precision_oot", row['Precision OOT'])
                mlflow.log_metric("recall_oot", row['Recall OOT'])
                mlflow.log_metric("f1_score_oot", row['F1-Score OOT'])
                mlflow.log_metric("accuracy_oot", row['Accuracy OOT'])
                mlflow.log_metric("objective", row['objective'])
            logger.info("Metrics logged to MLflow.")
    except Exception as e:
        logger.error(f"An error occurred during the MLflow run: {e}")
    finally:
        logger.info("MLflow run completed.")
    return results



def calculate_lift_statistics(y_true, y_probs, k_values=[0.01, 0.05, 0.1, 0.2, 0.5]):
    """
    Вычисляет lift statistics для заданных долей топ-к предсказаний.
    """
    n_total = len(y_true)
    n_positives = np.sum(y_true)
    baseline_rate = n_positives / n_total  # Общая доля положительных
    
    # Сортируем по убыванию вероятности
    sorted_indices = np.argsort(y_probs)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    lift_stats = {}
    for k in k_values:
        n_top = int(k * n_total)
        if n_top == 0:
            continue
        n_positives_top = np.sum(y_true_sorted[:n_top])
        top_rate = n_positives_top / n_top
        lift = top_rate / baseline_rate if baseline_rate > 0 else 0
        lift_stats[f'lift_top_{int(k*100)}'] = lift  # Убрали % для совместимости с MLflow
    
    return lift_stats