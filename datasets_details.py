import string
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import matplotlib.pyplot as plt
import plotly.io as pio

# Function to load SMS dataset
def SMS_ds():
    dataset_name = 'SMS'
    df = pd.read_csv("datasets/SMS_Spam.txt", sep='\t', names=['label', 'message'], encoding='latin-1') 
    df['label'].replace({"ham": 0, "spam": 1}, inplace=True) 
    df = df[['message', 'label']]
    return dataset_name, df

# Function to load Twitter dataset
def Twitter_ds():
    dataset_name = 'Twitter'
    df = pd.read_csv("datasets/Twitter_Spam.csv")
    df.drop(["Id", "following", "followers", "actions", "is_retweet", "location"], inplace=True, axis=1)
    df["Type"].replace({"Quality": 0, "Spam": 1}, inplace=True)
    df.rename({"Type": "label", "Tweet": "message"}, axis=1, inplace=True)
    return dataset_name, df

# Function to load YouTube dataset
def youTube_ds():
    dataset_name = 'YouTube'
    df = pd.read_csv("datasets/YouTube_Spam.csv", encoding='latin-1')
    df.drop(["COMMENT_ID", "AUTHOR", "DATE"], inplace=True, axis=1) 
    df.rename({"CLASS": "label", "CONTENT": "message"}, axis=1, inplace=True)
    return dataset_name, df

# Function to clean text data
def clean_dataset(sentence): 
    # Takes a sentence, removes punctuation, tokenizes, removes stopwords, and returns a cleaned sentence limited to 30 words.
    stop_words = set(stopwords.words('english'))
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(sentence)
    cleaned_sentence = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_sentence[:30])

# Function to plot histograms
def plot_histogram(df, dataset_name, column_name, title_prefix):
    # Creates a histogram using Plotly Express based on the length of words in a specified column of a DataFrame. It customizes the layout and displays the histogram.
    word_counts = df[column_name].apply(lambda x: len(word_tokenize(x)))

    fig = px.histogram(
        x=word_counts,
        nbins=max(word_counts) - min(word_counts) + 1,
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


def find_min_max_instance_length(df, column_index):
    # For a specified column in a DataFrame, this function finds the instance with the maximum and minimum lengths, along with their details. It returns a DataFrame with this information.
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



# Load datasets
SMS_dataset_name, df_sms = SMS_ds()
TW_dataset_name, df_twitter = Twitter_ds()
YT_dataset_name, df_youtube = youTube_ds()

# Apply preprocessing
df_youtube["cleaned_message"] = df_youtube["message"].apply(clean_dataset)
df_twitter["cleaned_message"] = df_twitter["message"].apply(clean_dataset)
df_sms["cleaned_message"] = df_sms["message"].apply(clean_dataset)

# Display results and plot histograms
# Prints information about instance lengths before and after preprocessing for each dataset.
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

# Plot histograms before preprocessing
plot_histogram(df_youtube, YT_dataset_name, 'message', 'before')
plot_histogram(df_twitter, TW_dataset_name, 'message', 'before')
plot_histogram(df_sms, SMS_dataset_name, 'message', 'before')

# Plot histograms after preprocessing
plot_histogram(df_youtube, YT_dataset_name, 'cleaned_message', 'after')
plot_histogram(df_twitter, TW_dataset_name, 'cleaned_message', 'after')
plot_histogram(df_sms, SMS_dataset_name, 'cleaned_message', 'after')
