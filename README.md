# Hate Speech Detection

## Introduction
This project aims to build a model that can detect hate speech in text. We use a dataset from Twitter and various Python libraries to preprocess the data, build the model, and evaluate its performance.

## Dataset
We use the `twitter.csv` dataset for this project. This dataset contains tweets that are labeled as hate speech or not.

## Libraries Used
- pandas: For data manipulation and analysis.
- re: To detect username, URL, and HTML entity.
- string: For string operations.
- nltk: Natural Language Toolkit, for text manipulation.
- stopwords from nltk.corpus: To remove stopwords.
- CountVectorizer from sklearn.feature_extraction.text: To convert text data to token count.
- train_test_split from sklearn.model_selection: To split the dataset into training and test sets.
- DecisionTreeClassifier from sklearn.tree: To build the Decision Tree model.
- confusion_matrix, accuracy_score from sklearn.metrics: For model evaluation.
- seaborn, matplotlib: For graphical representation of test output.

## Steps
1. **Importing Libraries**: Import all the necessary libraries mentioned above.
2. **Read the Data**: Load the `twitter.csv` dataset using pandas.
3. **Clean the Data**: Preprocess the data by removing unnecessary elements like usernames, URLs, HTML entities, and stopwords.
4. **Splitting the Dataset**: Split the dataset into a training set and a validation set using `train_test_split`.
5. **Building the Model**: Build a Decision Tree Classifier model using the training set.
6. **Evaluation**: Evaluate the model using a confusion matrix and accuracy score.
7. **Graphical Representation**: Visualize the test data results using seaborn and matplotlib.
8. **Test Run**: Run the model on the test data to predict hate speech.

## Conclusion
This project helps in understanding how machine learning can be used to detect hate speech in text. The performance of the Decision Tree model can be improved by tuning the parameters or using different models.
