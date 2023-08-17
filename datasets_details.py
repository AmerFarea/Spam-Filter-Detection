import string
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def SMS_ds():
    dataset_name = 'SMS'
    df = pd.read_csv("../SMS_Spam.txt", sep='\t', names=['label', 'message'], encoding='latin-1') 
    df['label'].replace({"ham": 0, "spam": 1}, inplace=True) 
    df = df[['message', 'label']]
    y = df['label']
    return dataset_name, df

def Twitter_ds():
    dataset_name = 'Twitter'
    df = pd.read_csv("../Twitter_Spam.csv")
    df.drop(["Id", "following", "followers", "actions", "is_retweet", "location"], inplace=True, axis=1)
    df["Type"].replace({"Quality": 0, "Spam": 1}, inplace=True)
    df.rename({"Type": "label", "Tweet": "message"}, axis=1, inplace=True)
    y = df['label']
    return dataset_name, df

def youTube_ds():
    dataset_name = 'YouTube'
    df = pd.read_csv("../YouTube_Spam.csv", encoding='latin-1')
    df.drop(["COMMENT_ID", "AUTHOR", "DATE"], inplace=True, axis=1) 
    df.rename({"CLASS": "label", "CONTENT": "message"}, axis=1, inplace=True)
    y = df['label']
    return dataset_name, df

# Load datasets
dataset_name_sms, df_sms = SMS_ds()
dataset_name_twitter, df_twitter = Twitter_ds()
dataset_name_youtube, df_youtube = youTube_ds()

# Preprocessing function
def clean_dataset(sentence): 
    stop_words = set(stopwords.words('english'))
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(sentence)
    cleaned_sentence = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_sentence[:30])

# Apply preprocessing
df_youtube["cleaned_message"] = df_youtube["message"].apply(clean_dataset)
df_twitter["cleaned_message"] = df_twitter["message"].apply(clean_dataset)
df_sms["cleaned_message"] = df_sms["message"].apply(clean_dataset)

# Display information
print("YouTube Dataset Info. before preprocessing:")
print(min_max_instance_length_before_processing(df_youtube))
print("YouTube Dataset Info. after preprocessing:")
print(min_max_instance_length_after_processing(df_youtube))

print("Twitter Dataset Info. before preprocessing:")
print(min_max_instance_length_before_processing(df_twitter))
print("Twitter Dataset Info. after preprocessing:")
print(min_max_instance_length_after_processing(df_twitter))

print("SMS Dataset Info. before preprocessing:")
print(min_max_instance_length_before_processing(df_sms))
print("SMS Dataset Info. after preprocessing:")
print(min_max_instance_length_after_processing(df_sms))


def find_min_max_instance_length(df, column_index):
    max_length = 0
    max_cell = None

    min_length = float('inf')
    min_cell = None

    column_to_check = df.columns[column_index]

    max_results = []
    min_results = []

    for index, row in df.iterrows():
        cell_content = str(row[column_to_check])
        cell_length = len(cell_content)
        
        if cell_length > max_length:
            max_length = cell_length
            max_cell = (index, column_to_check, cell_content)
        
        if cell_length < min_length:
            min_length = cell_length
            min_cell = (index, column_to_check, cell_content)

    if max_cell:
        max_row, max_column, max_value = max_cell
        max_results.append({'Type': 'Max', 'Row': max_row, 'Column': max_column, 'Value': max_value, 'Length': max_length})
    else:
        max_results.append({'Type': 'Max', 'Row': 'N/A', 'Column': 'N/A', 'Value': 'N/A', 'Length': 'N/A'})

    if min_cell:
        min_row, min_column, min_value = min_cell
        min_results.append({'Type': 'Min', 'Row': min_row, 'Column': min_column, 'Value': min_value, 'Length': min_length})
    else:
        min_results.append({'Type': 'Min', 'Row': 'N/A', 'Column': 'N/A', 'Value': 'N/A', 'Length': 'N/A'})

    max_df = pd.DataFrame(max_results)
    min_df = pd.DataFrame(min_results)

    combined_df = pd.concat([max_df, min_df], ignore_index=True)

    return combined_df

def clean_dataset(sentence): 
    stop_words = set(stopwords.words('english'))
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(sentence)
    cleaned_sentence = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_sentence[:30])

# Load datasets
SMS_dataset_name, df_sms = SMS_ds()
TW_dataset_name, df_twitter = Twitter_ds()
YT_dataset_name, df_youtube = youTube_ds()

# Apply preprocessing
df_youtube["cleaned_message"] = df_youtube["message"].apply(clean_dataset)
df_twitter["cleaned_message"] = df_twitter["message"].apply(clean_dataset)
df_sms["cleaned_message"] = df_sms["message"].apply(clean_dataset)

# Display results
print("YouTube Dataset Info. before preprocessing:")
print(find_min_max_instance_length(df_youtube, 0))
print("YouTube Dataset Info. after preprocessing:")
print(find_min_max_instance_length(df_youtube, 2))

print("\nTwitter Dataset Info. before preprocessing:")
print(find_min_max_instance_length(df_twitter, 0))
print("Twitter Dataset Info. after preprocessing:")
print(find_min_max_instance_length(df_twitter, 2))

print("\nSMS Dataset Info. before preprocessing:")
print(find_min_max_instance_length(df_sms, 0))
print("SMS Dataset Info. after preprocessing:")
print(find_min_max_instance_length(df_sms, 2))



def plot_histogram(df, dataset_name, column_name, title_prefix):
    word_counts = []
    for cell_content in df[column_name]:
        words = word_tokenize(cell_content)
        word_counts.append(len(words))

    fig = px.histogram(
        x=word_counts,
        nbins=max(word_counts) - min(word_counts) + 0,
        title=f"The distribution of the {dataset_name} dataset based on the number of words in each instance {title_prefix} preprocessing",
        labels={"x": "Instance Length (Number of Words)", "y": "Frequency"}
    )

    fig.update_layout(
        xaxis_title="Instance Length (Number of Words)",
        yaxis_title="Frequency",
        yaxis=dict(type='linear'),
        title_font_size=20,
        font=dict(size=14),
        width=1200,
        height=500,
    )

    fig.update_layout(xaxis=dict(tickmode='linear', tick0=min(word_counts), dtick=10))
    fig.show()

def plot_roc_auc(models, roc_auc_values):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for model, roc_auc in zip(models, roc_auc_values):
        plt.plot(roc_auc, label=f'{model} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

# Load datasets
SMS_dataset_name, df_sms = SMS_ds()
TW_dataset_name, df_twitter = Twitter_ds()
YT_dataset_name, df_youtube = youTube_ds()

# Histograms before preprocessing
plot_histogram(df_youtube, YT_dataset_name, 'message', 'before')
plot_histogram(df_twitter, TW_dataset_name, 'message', 'before')
plot_histogram(df_sms, SMS_dataset_name, 'message', 'before')

# Histograms after preprocessing
plot_histogram(df_youtube, YT_dataset_name, 'cleaned_message', 'after')
plot_histogram(df_twitter, TW_dataset_name, 'cleaned_message', 'after')
plot_histogram(df_sms, SMS_dataset_name, 'cleaned_message', 'after')

# ROC-AUC plot
models = ['RoBERTa', 'DeRERTa', 'BART']
roc_auc_values = [0.98497, 0.98023, 0.97921]
plot_roc_auc(models, roc_auc_values)

