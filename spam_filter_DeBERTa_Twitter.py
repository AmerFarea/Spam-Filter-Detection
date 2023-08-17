import os
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from transformers import DebertaTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')

# Reading Twitter dataset
df = pd.read_csv("/content/drive/MyDrive//Twitter_Spam.csv")
df.drop(["Id", "following", "followers", "actions", "is_retweet", "location"], inplace=True, axis=1)
df["Type"].replace({"Quality": 0, "Spam":1}, inplace=True)
df.rename({"Type": "label", "Tweet": "message"}, axis=1, inplace=True)
y = df['label']
dataset_name = "Twitter"

# Preprocessing
def clean_sentence(sentence):
    stop_words = set(stopwords.words('english'))
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(sentence)
    cleaned_sentence = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_sentence[:30])

df["cleaned_message"] = df["message"].apply(clean_sentence)

# Loading DeBERTa model
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-large")
model = TFAutoModel.from_pretrained("microsoft/deberta-large")
model_name = "DeBERTa"

# Embedding messages
all_list = []
for i in df["cleaned_message"]:
    all_list.append(i)


max_length = 32
inputs = tokenizer(all_list, return_tensors="tf", padding=True, truncation=True, max_length=max_length)
embedded_messages = model(inputs)[0].numpy()
df_embedded = pd.DataFrame(embedded_messages[:, 0, :])

# Splitting data
# Here if the ratio of the test_size is not specifyied , it will be be default 25%
X_train, X_test, y_train, y_test = train_test_split(df_embedded, y, stratify=y, random_state=3)

# Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_shape=(1024,), activation="relu"),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compiling the model
Init_LR = 0.001
Bs = 256
opt = tf.keras.optimizers.RMSprop(learning_rate=Init_LR)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.000001)

# Training the model
history = model.fit(X_train, y_train, validation_split=0.2, batch_size=Bs, epochs=50, callbacks=[learning_rate_reduction])

# Plot and save learning rate
def plot_and_save_learning_curve(history, filename):
    lr = history.history['lr']
    loss = history.history['loss']

    with open(filename.replace('_plot.pdf', '.txt'), "w") as f:
        f.write("lr\tloss\n")
        for lr_val, loss_val in zip(lr, loss):
            f.write(f"{lr_val}\t{loss_val}\n")

    plt.plot(lr, loss)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Schedule')
    plt.savefig(filename, format='pdf')
    plt.show()

plot_and_save_learning_curve(history, f"/content/drive/MyDrive/{model_name}_{dataset_name}_learning_curve_plot.pdf")

# Save training history
history_filename = f"{model_name}_{dataset_name}_history.txt"
with open(history_filename, "w") as txt_file:
    txt_file.write("Epoch\tLoss\tAccuracy\tVal Loss\tVal Accuracy\n")
    for epoch, metrics in enumerate(history.history['accuracy']):
        txt_file.write(f"{epoch+1}\t{history.history['loss'][epoch]:.4f}\t{history.history['accuracy'][epoch]:.4f}\t"
                       f"{history.history['val_loss'][epoch]:.4f}\t{history.history['val_accuracy'][epoch]:.4f}\n")

print("Training history details saved to", history_filename)

# Predictions and evaluation
preds = np.round(model.predict(X_test)).flatten()

# Plot confusion matrix
plt.figure(figsize=(5, 5))
heatmap = sns.heatmap(data=pd.DataFrame(confusion_matrix(y_test, preds)), annot=True, fmt="d", cmap=sns.color_palette("Reds", 50))
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.savefig(f"/content/drive/MyDrive/{model_name}_{dataset_name}_confusion_matrix.pdf")
plt.close()

# Print accuracy, ROC and classification report
accuracy = round(accuracy_score(y_test, preds), 5) * 100
roc_auc = round(roc_auc_score(y_test, preds), 5)
classification_rep = classification_report(y_test, preds)

print(f"""Accuracy: {accuracy}%
ROC-AUC: {roc_auc}""")
print(classification_rep)

with open(f"/content/drive/MyDrive/{model_name}_{dataset_name}_results.txt", "w") as txt_file:
    txt_file.write(f"Accuracy: {accuracy}%\n")
    txt_file.write(f"ROC-AUC: {roc_auc}\n")
    txt_file.write(classification_rep)

with open(f"/content/drive/MyDrive/{model_name}_{dataset_name}_roc_auc.txt", "w") as f:
    f.write("y_test\tpreds\n")
    for true_label, pred_label in zip(y_test, preds):
        f.write(f"{true_label}\t{int(pred_label)}\n")
