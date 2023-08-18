# This is a spam filter detection repo

## spam_filter.py :

This code performs text classification using different NLP and machine learning models (i.e., `BART`, `RoBERta`, and `DeBERTa`) on three different datasets: `SMS`, `Twitter`, and `YouTube`. The code also includes data preprocessing, model training, evaluation, and result visualization. Here's a detailed description of each section of the code:

1. **Importing Libraries**: The code starts by importing various Python libraries including `os`, `string`, `numpy`, `pandas`, `tensorflow`, `nltk`, `transformers`, `sklearn`, `seaborn`, and `matplotlib`. These libraries are used for data manipulation, natural language processing, machine learning, and visualization.

2. **Data Loading and Preprocessing**:

   - SMS Dataset: The SMS dataset is loaded from a text file and preprocessed. The 'ham' and 'spam' labels are converted to binary values (0 and 1) for classification.
   - Twitter Dataset: The Twitter dataset is loaded and unnecessary columns are dropped. The 'Quality' and 'Spam' labels are converted to binary values.
   - YouTube Dataset: The YouTube dataset is loaded and unnecessary columns are dropped. The 'CLASS' labels are used for binary classification.
   - Text Preprocessing: A function `clean_sentence()` is defined to clean and tokenize sentences. Stop words and punctuation are removed, and tokenized words are cleaned.

3. **Model Loading and Tokenization**:

   - The `load_model_and_tokenizer()` function is defined to load the chosen transformer-based models and their corresponding tokenizers.
   - The desired dataset and model are selected (in this case, the YouTube dataset and BART model).

4. **Embedding Messages**:

   - The cleaned sentences from the selected dataset are tokenized and embedded using the chosen transformer model.
   - The resulting embedded messages are stored in a DataFrame.

5. **Data Splitting**:

   - The embedded messages and their corresponding labels are split into training and testing sets using the `train_test_split()` function.

6. **Model Architecture**:

   - A feedforward neural network model is defined using TensorFlow's `Sequential` API.
   - The model consists of several dense layers with batch normalization and dropout to prevent overfitting.

7. **Model Compilation**:

   - The model is compiled with an optimizer, learning rate, loss function, and evaluation metrics.

8. **Model Training**:

   - The model is trained using the training data. A learning rate reduction callback is applied to adjust the learning rate during training.

9. **Training History and Results**:

   - The training history, including loss and accuracy metrics for each epoch, is saved to a text file.
   - Model predictions are made on the testing data, and the results are evaluated using metrics such as accuracy, ROC-AUC, and classification report.

10. **Confusion Matrix Visualization**:

- A confusion matrix is generated based on the true labels and predicted labels of the testing data.
- The confusion matrix is visualized using a heatmap and saved as a PDF file.

11. **Results Reporting**:

- The accuracy, ROC-AUC, and classification report are printed to the console.
- The same results are saved to a text file for reference.

12. **ROC-AUC Data Storage**:

- The true labels and predicted labels are saved to a file for further ROC-AUC analysis.

13. **Learning Curve Visualization**:

- A function `plot_and_save_learning_curve()` is defined to plot and save the learning curve of the model.

Overall, this code provides a comprehensive pipeline for loading, preprocessing, training, evaluating, and visualizing the results of text classification models on different datasets using transformer-based models. The selected dataset, model, and parameters can be customized as needed.
