import os
import string # for string
import numpy as np # for mathmatical operation
import pandas as pd # for reading the data
import tensorflow as tf
from nltk.corpus import stopwords # import stop words
from nltk.tokenize import word_tokenize # for spliting the sentences
import nltk
nltk.download('stopwords')
nltk.download('punkt')



"""## Reading SMS dataset"""

# df = pd.read_csv("/content/drive/MyDrive/SMS_Spam.txt", sep='\t' , names=['label', 'message'], encoding='latin-1')
# df['label'].replace({"ham": 0, "spam":1}, inplace=True)
# y = df['label']
# dataset_name = "SMS"

"""## Reading Twitter dataset"""

df = pd.read_csv("/content/drive/MyDrive//Twitter_Spam.csv") # reading the data using read_csv function
df.drop(["Id", "following", "followers", "actions", "is_retweet", "location"], inplace=True, axis=1)
df["Type"].replace({"Quality": 0, "Spam":1}, inplace=True)
df.rename({"Type": "label", "Tweet": "message"},axis=1, inplace=True)
y = df['label']
dataset_name = "Twitter"

"""## Reading YouTube dataset"""

# df = pd.read_csv("/content/drive/MyDrive//YouTube_Spam.csv", encoding='latin-1')
# df.drop(["COMMENT_ID", "AUTHOR", "DATE"], inplace=True, axis=1)
# df.rename({"CLASS": "label", "CONTENT": "message"},axis=1, inplace=True)
# y = df['label']
# dataset_name = "YouTube"
"""## Preprocessing"""

def clean_sentence(sentence):
    stop_words = set(stopwords.words('english')) # load stop words into stop_words
    sentence = sentence.translate(str.maketrans('','',string.punctuation)) # remove punctuation
    tokens = word_tokenize(sentence) # tokenize the sentences
    cleaned_sentence = [word for word in tokens if word not in stop_words] # remove stop words
    return " ".join(cleaned_sentence[:30]) # using the first 30 tokens only

df["cleaned_message"] = df["message"].apply(clean_sentence) # applying the clean_sentence() funiction to each row

"""## Loading RoBERTa model"""

# from transformers import AutoTokenizer, TFRobertaModel
# import tensorflow as tf

# tokenizer = AutoTokenizer.from_pretrained("roberta-large")
# model = TFRobertaModel.from_pretrained("roberta-large")
# model_name = "RoBERTa"

"""## Loading DeBERTa model"""
import tensorflow as tf
from transformers import DebertaTokenizer, TFAutoModel

tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-large")
model = TFAutoModel.from_pretrained("microsoft/deberta-large")
model_name = "DeBERTa"





"""## Loading BART model"""
# from transformers import AutoTokenizer, TFBartModel # import the model and the tokenizer
# import tensorflow as tf # import tensorflow
# model_name = "BART"

# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
# model = TFBartModel.from_pretrained("facebook/bart-large")



####

all_list = [] # make a list contains all messages
for i in df["cleaned_message"]:
  all_list.append(i)


max_length = 32
inputs = tokenizer(all_list, return_tensors="tf", padding=True, truncation=True, max_length=max_length)
# inputs = tokenizer(all_list, return_tensors="tf", padding=True, truncation=True)

# Process inputs using the model (on GPU)
# with tf.device("/device:GPU:0"):  # Use /device:GPU:0 or the appropriate GPU device
#     outputs = model(inputs)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             outputs = model(inputs)
#     except RuntimeError as e:
#         print(e)


# outputs = model(inputs)



import tensorflow as tf
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)
strategy = tf.distribute.TPUStrategy(resolver)


# Inside the strategy scope, your computations will be executed on the TPU
with strategy.scope():
    outputs = model(inputs)






# Convert model outputs to a numpy array (on GPU)
embedded_messages = outputs[0].numpy()

# Create a DataFrame from the embedded messages
df_embedded = pd.DataFrame(embedded_messages[:, 0, :])


from sklearn.model_selection import train_test_split # importing train_test_split from sklearn.model_selection to split the data
X_train, X_test, y_train, y_test = train_test_split(df_embedded, y, stratify=y, random_state=3) # start spliting the data

import tensorflow as tf # importing tensorflow
from tensorflow.keras.models import Sequential # import the sequential model
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Adagrad
from tensorflow.keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Dense(512, input_shape=(1024,), activation="relu"))
model.add(BatchNormalization(axis=-1))

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

model.add(Dense(128, activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

model.add(Dense(100, activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

model.add(Dense(1, activation="sigmoid"))

Init_LR = 0.001 # initializing the learning rate
Num_Epochs = 50 # setting number of epochs
Bs = 256 # setting the batch size
#opt = SGD(learning_rate=Init_LR, nesterov=True)
#opt = Adam(learning_rate=Init_LR)
#opt = Adagrad(learning_rate=Init_LR)
opt = RMSprop(learning_rate = Init_LR) # setig the optimizer
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"]) # start compiling the model
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.000001) # setting learning rate reduction parameters



history = model.fit(X_train, y_train, validation_split=0.2, batch_size=Bs, epochs=Num_Epochs, callbacks=[learning_rate_reduction]) # start fitting the model


# Save the printed values to a text file
# model_name = "BART"
# dataset_name = "YouTube"
history_filename = f"{model_name}_{dataset_name}_history.txt"

with open(history_filename, "w") as txt_file:
    txt_file.write("Epoch\tLoss\tAccuracy\tVal Loss\tVal Accuracy\n")
    for epoch, metrics in enumerate(history.history['accuracy']):
        txt_file.write(f"{epoch+1}\t{history.history['loss'][epoch]:.4f}\t{history.history['accuracy'][epoch]:.4f}\t"
                       f"{history.history['val_loss'][epoch]:.4f}\t{history.history['val_accuracy'][epoch]:.4f}\n")

print("Training history details saved to", history_filename)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# predict `preds` and prepare the data
preds = np.round(model.predict(X_test)).flatten()
# Plot confusion matrix
plt.figure(figsize=(5, 5))
heatmap = sns.heatmap(data=pd.DataFrame(confusion_matrix(y_test, preds)), annot=True, fmt="d", cmap=sns.color_palette("Reds", 50))
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.savefig(f"/content/drive/MyDrive/{model_name}_{dataset_name}_confusion_matrix.pdf")  # Save the plot as a PDF file
plt.close()  # Close the plot to free up memory

# Print accuracy, ROC and classification report for the test-set
accuracy = round(accuracy_score(y_test, preds), 5) * 100
roc_auc = round(roc_auc_score(y_test, preds), 5)
classification_rep = classification_report(y_test, preds)

print(f"""Accuray: {accuracy}%
ROC-AUC: {roc_auc}""")
print(classification_rep)

with open(f"/content/drive/MyDrive/{model_name}_{dataset_name}_results.txt", "w") as txt_file:
    txt_file.write(f"Accuray: {accuracy}%\n")
    txt_file.write(f"ROC-AUC: {roc_auc}\n")
    txt_file.write(classification_rep)


with open(f"/content/drive/MyDrive/{model_name}_{dataset_name}_roc_auc.txt", "w") as f:
    f.write("y_test\tpreds\n")  # Write header
    for true_label, pred_label in zip(y_test, preds):
        f.write(f"{true_label}\t{int(pred_label)}\n")




##### plotting the learning curve #####

def plot_and_save_learning_rate(history, filename):
    # Extract the learning rates and losses from the history object
    lr = history.history['lr']
    loss = history.history['loss']

    with open(f"/content/drive/MyDrive/{model_name}_{dataset_name}_learning_rate.txt", "w") as f:
      f.write("lr\tloss\n")  # Write header
      for true_label, pred_label in zip(lr, loss):
          f.write(f"{lr}\t{loss}\n")

    # Plot the learning rate over epochs
    plt.plot(lr, loss)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Schedule')

    # Save the plot as a PDF file
    plt.savefig(filename, format='pdf')
    plt.show()

# Assuming `history` is the result of model.fit(...)
plot_and_save_learning_rate(history, '/content/drive/MyDrive/{model_name}_{dataset_name}_learning_rate_plot.pdf')
