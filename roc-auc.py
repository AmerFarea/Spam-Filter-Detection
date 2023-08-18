from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

def calculate_roc_auc(data_path):
    df = pd.read_csv(data_path, delimiter='\t')
    y_test = df.iloc[:, 0:1]
    preds = df.iloc[:, 1:]
    fpr, tpr, _ = roc_curve(y_test, preds)
    auc_value = auc(fpr, tpr)
    return fpr, tpr, auc_value

def plot_roc_auc(models, roc_auc_values, tpr_values, fpr_values, model_name, dataset_name):
    plt.figure(figsize=(8, 6))
    for i, model in enumerate(models):
        plt.plot(fpr_values[i], tpr_values[i], marker='o', label=f'{model} (AUC = {roc_auc_values[i]:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC-AUC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"roc-auc-data-and-plots/{model_name}_{dataset_name}_roc_auc.pdf")
    plt.close()

# Common data
model_name = 'BART_RoBERTa_RoBERTa'
dataset_names = ['sms', 'twitter', 'youtube']
model_names = ['BART', 'DeBERTa', 'RoBERTa']

for dataset_name in dataset_names:
    fpr_values, tpr_values, auc_values = [], [], []

    for model in model_names:
        data_path = f'roc-auc-data-and-plots/{model}_{dataset_name}_roc_auc.txt'
        fpr, tpr, auc_value = calculate_roc_auc(data_path)
        fpr_values.append(fpr)
        tpr_values.append(tpr)
        auc_values.append(auc_value)

    plot_roc_auc(model_names, auc_values, tpr_values, fpr_values, model_name, dataset_name)