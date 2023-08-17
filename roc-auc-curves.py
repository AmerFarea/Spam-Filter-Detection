
############################################################################### ROC-AUC CODE for YouTube dataset ################################################################################

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd


model_name = 'BART_RoBERTa_RoBERTa'
dataset_name = 'YouTube'

# Read data for BART model
df_bart_youtube = pd.read_csv('BART_YouTube_roc_auc.txt', delimiter='\t')
y_test_bart_youtube = df_bart_youtube.iloc[:, 0:1]
preds_bart_youtube = df_bart_youtube.iloc[:, 1:]

# Calculate ROC-AUC for BART model
bart_youtube_fpr, bart_youtube_tpr, _ = roc_curve(y_test_bart_youtube, preds_bart_youtube)
auc_bart_youtube = auc(bart_youtube_fpr, bart_youtube_tpr)



# Read data for DeBERTa model
df_DeBERTa_youtube = pd.read_csv('DeBERTa_YouTube_roc_auc.txt', delimiter='\t')
y_test_DeBERTa_youtube = df_DeBERTa_youtube.iloc[:, 0:1]
preds_DeBERTa_youtube = df_DeBERTa_youtube.iloc[:, 1:]

# Calculate ROC-AUC for DeBERTa model
DeBERTa_youtube_fpr, DeBERTa_youtube_tpr, _ = roc_curve(y_test_DeBERTa_youtube, preds_DeBERTa_youtube)
auc_DeBERTa_youtube = auc(DeBERTa_youtube_fpr, DeBERTa_youtube_tpr)




# Read data for RoBERTa model
df_RoBERTa_youtube = pd.read_csv('RoBERTa_YouTube_roc_auc.txt', delimiter='\t')
y_test_RoBERTa_youtube = df_RoBERTa_youtube.iloc[:, 0:1]
preds_RoBERTa_youtube = df_RoBERTa_youtube.iloc[:, 1:]

# Calculate ROC-AUC for RoBERTa model
RoBERTa_youtube_fpr, RoBERTa_youtube_tpr, _ = roc_curve(y_test_RoBERTa_youtube, preds_RoBERTa_youtube)
auc_RoBERTa_youtube = auc(RoBERTa_youtube_fpr, RoBERTa_youtube_tpr)




# Given data
models = ['BART', 'RoBERTa', 'RoBERTa']
roc_auc_values = [auc_bart_youtube, auc_DeBERTa_youtube, auc_RoBERTa_youtube]
tpr_values = [bart_youtube_tpr, DeBERTa_youtube_tpr,  RoBERTa_youtube_tpr]
fpr_values = [bart_youtube_fpr, DeBERTa_youtube_fpr,  RoBERTa_youtube_fpr]

# Plotting the ROC-AUC curve
plt.figure(figsize=(8, 6))
for i, model in enumerate(models):
    plt.plot(fpr_values[i], tpr_values[i], marker='o', label=f'{model} (AUC = {roc_auc_values[i]:.4f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC-AUC Curve')
plt.xlabel('False Positive Rate  -->')
plt.ylabel('True Positive Rate  -->')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Display the plot

plt.savefig(f"{model_name}_{dataset_name}_roc_auc.pdf")  # Save the plot as a PDF file
plt.close()  # Close the plot to free up memory
plt.show()

############################################################################### ROC-AUC CODE for SMS dataset ################################################################################

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd


model_name = 'BART_RoBERTa_RoBERTa'
dataset_name = 'sms'

# Read data for BART model
df_bart_sms = pd.read_csv('BART_sms_roc_auc.txt', delimiter='\t')
y_test_bart_sms = df_bart_sms.iloc[:, 0:1]
preds_bart_sms = df_bart_sms.iloc[:, 1:]

# Calculate ROC-AUC for BART model
bart_sms_fpr, bart_sms_tpr, _ = roc_curve(y_test_bart_sms, preds_bart_sms)
auc_bart_sms = auc(bart_sms_fpr, bart_sms_tpr)



# Read data for DeBERTa model
df_DeBERTa_sms = pd.read_csv('DeBERTa_sms_roc_auc.txt', delimiter='\t')
y_test_DeBERTa_sms = df_DeBERTa_sms.iloc[:, 0:1]
preds_DeBERTa_sms = df_DeBERTa_sms.iloc[:, 1:]

# Calculate ROC-AUC for DeBERTa model
DeBERTa_sms_fpr, DeBERTa_sms_tpr, _ = roc_curve(y_test_DeBERTa_sms, preds_DeBERTa_sms)
auc_DeBERTa_sms = auc(DeBERTa_sms_fpr, DeBERTa_sms_tpr)




# Read data for RoBERTa model
df_RoBERTa_sms = pd.read_csv('RoBERTa_sms_roc_auc.txt', delimiter='\t')
y_test_RoBERTa_sms = df_RoBERTa_sms.iloc[:, 0:1]
preds_RoBERTa_sms = df_RoBERTa_sms.iloc[:, 1:]

# Calculate ROC-AUC for RoBERTa model
RoBERTa_sms_fpr, RoBERTa_sms_tpr, _ = roc_curve(y_test_RoBERTa_sms, preds_RoBERTa_sms)
auc_RoBERTa_sms = auc(RoBERTa_sms_fpr, RoBERTa_sms_tpr)




# Given data
models = ['BART', 'RoBERTa', 'RoBERTa']
roc_auc_values = [auc_bart_sms, auc_DeBERTa_sms, auc_RoBERTa_sms]
tpr_values = [bart_sms_tpr, DeBERTa_sms_tpr,  RoBERTa_sms_tpr]
fpr_values = [bart_sms_fpr, DeBERTa_sms_fpr,  RoBERTa_sms_fpr]

# Plotting the ROC-AUC curve
plt.figure(figsize=(8, 6))
for i, model in enumerate(models):
    plt.plot(fpr_values[i], tpr_values[i], marker='o', label=f'{model} (AUC = {roc_auc_values[i]:.4f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC-AUC Curve')
plt.xlabel('False Positive Rate  -->')
plt.ylabel('True Positive Rate  -->')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Display the plot

plt.savefig(f"{model_name}_{dataset_name}_roc_auc.pdf")  # Save the plot as a PDF file
plt.close()  # Close the plot to free up memory
plt.show()

############################################################################### ROC-AUC CODE for Twitter ################################################################################

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd


model_name = 'BART_RoBERTa_RoBERTa'
dataset_name = 'twitter'

# Read data for BART model
df_bart_twitter = pd.read_csv('BART_twitter_roc_auc.txt', delimiter='\t')
y_test_bart_twitter = df_bart_twitter.iloc[:, 0:1]
preds_bart_twitter = df_bart_twitter.iloc[:, 1:]

# Calculate ROC-AUC for BART model
bart_twitter_fpr, bart_twitter_tpr, _ = roc_curve(y_test_bart_twitter, preds_bart_twitter)
auc_bart_twitter = auc(bart_twitter_fpr, bart_twitter_tpr)



# Read data for DeBERTa model
df_DeBERTa_twitter = pd.read_csv('DeBERTa_twitter_roc_auc.txt', delimiter='\t')
y_test_DeBERTa_twitter = df_DeBERTa_twitter.iloc[:, 0:1]
preds_DeBERTa_twitter = df_DeBERTa_twitter.iloc[:, 1:]

# Calculate ROC-AUC for DeBERTa model
DeBERTa_twitter_fpr, DeBERTa_twitter_tpr, _ = roc_curve(y_test_DeBERTa_twitter, preds_DeBERTa_twitter)
auc_DeBERTa_twitter = auc(DeBERTa_twitter_fpr, DeBERTa_twitter_tpr)




# Read data for RoBERTa model
df_RoBERTa_twitter = pd.read_csv('RoBERTa_twitter_roc_auc.txt', delimiter='\t')
y_test_RoBERTa_twitter = df_RoBERTa_twitter.iloc[:, 0:1]
preds_RoBERTa_twitter = df_RoBERTa_twitter.iloc[:, 1:]

# Calculate ROC-AUC for RoBERTa model
RoBERTa_twitter_fpr, RoBERTa_twitter_tpr, _ = roc_curve(y_test_RoBERTa_twitter, preds_RoBERTa_twitter)
auc_RoBERTa_twitter = auc(RoBERTa_twitter_fpr, RoBERTa_twitter_tpr)




# Given data
models = ['BART', 'RoBERTa', 'RoBERTa']
roc_auc_values = [auc_bart_twitter, auc_DeBERTa_twitter, auc_RoBERTa_twitter]
tpr_values = [bart_twitter_tpr, DeBERTa_twitter_tpr,  RoBERTa_twitter_tpr]
fpr_values = [bart_twitter_fpr, DeBERTa_twitter_fpr,  RoBERTa_twitter_fpr]

# Plotting the ROC-AUC curve
plt.figure(figsize=(8, 6))
for i, model in enumerate(models):
    plt.plot(fpr_values[i], tpr_values[i], marker='o', label=f'{model} (AUC = {roc_auc_values[i]:.4f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC-AUC Curve')
plt.xlabel('False Positive Rate  -->')
plt.ylabel('True Positive Rate  -->')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Display the plot

plt.savefig(f"{model_name}_{dataset_name}_roc_auc.pdf")  # Save the plot as a PDF file
plt.close()  # Close the plot to free up memory
plt.show()
