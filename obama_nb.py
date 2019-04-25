import pandas as pd
import re

from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from pandas.core.common import SettingWithCopyWarning
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn import model_selection as model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from openpyxl.workbook import Workbook
import matplotlib.pyplot as plot
from sklearn.utils import shuffle
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

import warnings

# negative - 1968
# neutral - 1978
# positive - 1679

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

# Reading the excel file
df = pd.read_excel('trainingObamaRomneytweets.xlsx', sheet_name='Obama')
df.rename(columns={'Unnamed: 0': 'exampleid'}, inplace=True)

stop_words = set(stopwords.words('english'))

word_list = []
label_list = []
for line in df.index:
    tweet = str(df['Anootated tweet'][line]).lower()
    if df['Class'][line] != "irrelevant" and df['Class'][line] != "irrevelant" and type(df['Class'][line]) != float:
        c = int(df['Class'][line])
    else:
        c = df['Class'][line]
    tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)
    if '<e>' in tweet:
        tweet = re.sub('<e>.*?</e>', '', tweet)
    if '<a>' in tweet:
        content = re.compile('<a>(.*?)</a>', flags=re.DOTALL).findall(tweet)
        tweet = re.sub('<a>.*?</a>', str(content), tweet)
    tweet = re.sub('[^a-zA-Z\s]+', ' ', tweet)
    word_list.append(tweet.split())
    label_list.append(c)


processed_list = []
for comment in word_list:
    str = ""
    for word in comment:
        if len(word)>2 and word not in stop_words:
            str = str + " " + word
    str = str.lstrip()
    str = str.rstrip()
    str.replace(' ','')
    processed_list.append(str)

final_tweets ={'Tweet': processed_list, 'Class': label_list}
data_frame = pd.DataFrame(final_tweets, dtype=object)
data_frame = data_frame[data_frame.Class != 2]
data_frame = data_frame[data_frame.Class != "irrelevant"]
data_frame = data_frame[data_frame.Class != "irrevelant"]
data_frame = data_frame[data_frame.Tweet != " "]
data_frame = data_frame[data_frame.Class != " "]
data_frame = data_frame[data_frame.Tweet != ""]
data_frame = data_frame[data_frame.Class != ""]
data_frame = data_frame.dropna()

# positive_class = data_frame[data_frame.Class == 1]
# negative_class = data_frame[data_frame.Class == -1]
# neutral_class = data_frame[data_frame.Class == 0]
#
# positive_upsampled = resample(positive_class, replace=True, n_samples=len(negative_class), random_state=27)
# upsampled_frame = pd.concat([positive_upsampled, negative_class, neutral_class])
data_frame = shuffle(data_frame)

def class_accuracy(y_actual, y_pred):
    positive_counts_actual = 0
    positive_counts_predicted = 0
    negative_counts_actual = 0
    negative_counts_predicted = 0
    neutral_counts_actual = 0
    neutral_counts_predicted = 0

    for index,c in enumerate(y_actual):
        if c == 1:
            positive_counts_actual = positive_counts_actual + 1
            if y_pred[index] == 1:
                positive_counts_predicted = positive_counts_predicted + 1
        elif c == 0:
            neutral_counts_actual = neutral_counts_actual + 1
            if y_pred[index] == 0:
                neutral_counts_predicted = neutral_counts_predicted + 1
        elif c == -1:
            negative_counts_actual = negative_counts_actual + 1
            if y_pred[index] == -1:
                negative_counts_predicted = negative_counts_predicted + 1
    positive_accuracy = positive_counts_predicted / positive_counts_actual
    negative_accuracy = negative_counts_predicted / negative_counts_actual
    neutral_accuracy = neutral_counts_predicted / neutral_counts_actual
    total_accuaracy = positive_accuracy + negative_accuracy + neutral_accuracy
    # average_accuracy = (positive_accuracy + negative_accuracy + neutral_accuracy) / 3
    print("Positive accuracies")
    print(positive_accuracy)
    print("negative accuracies")
    print(negative_accuracy)
    print("neutral accuracies")
    print(neutral_accuracy)
    average_accuracy = accuracy_score(y_actual, y_pred, normalize=True)
    print("avg accuracy")
    print(average_accuracy)
    return average_accuracy


tweets = data_frame['Tweet']
tweets_after = tweets.tolist()
y = data_frame['Class']
y_class = y.tolist()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tweets_after)

smt = SMOTE()
X, y_class = smt.fit_sample(X, y_class)

final_tweets_upsampling ={'Tweet': X, 'Class': y_class}
data_frame_upsampled = pd.DataFrame(final_tweets_upsampling, dtype=object)

final_accuracies = []
# classifier
clf = MultinomialNB()
predictions = model.cross_val_predict(clf, X, y_class, cv=10)
print("**************** Naive Bayes Classifier *********************")
nb_avg_accuracy = class_accuracy(y_class, predictions)
final_accuracies.append(nb_avg_accuracy)
target_names = ['Negative','Neutral','positive']
print(classification_report(y_class,predictions, target_names=target_names))

clf = svm.SVC(kernel='rbf', gamma=0.58, C=0.81,class_weight='balanced')
predictions_svm = model.cross_val_predict(clf, X, y_class, cv=10)
print("********************SVM Classifier***************************")
svm_avg_accuracy = class_accuracy(y_class, predictions_svm)
final_accuracies.append(svm_avg_accuracy)
target_names = ['Negative','Neutral','positive']
print(classification_report(y_class,predictions_svm, target_names=target_names))


clf = DecisionTreeClassifier(random_state=0,class_weight='balanced')
predictions_decision_tree = model.cross_val_predict(clf, X, y_class, cv=10)
print("*************************Decision Tree***************************")
dt_avg_accuracy = class_accuracy(y_class, predictions_decision_tree)
final_accuracies.append(dt_avg_accuracy)
target_names = ['Negative','Neutral','positive']
print(classification_report(y_class,predictions_decision_tree, target_names=target_names))

clf = LogisticRegression()
predictions_logistic_regression = model.cross_val_predict(clf, X, y_class, cv=10)
print("****************************Logistic Regression*************************")
lr_avg_accuracy = class_accuracy(y_class, predictions_logistic_regression)
final_accuracies.append(lr_avg_accuracy)
target_names = ['Negative','Neutral','positive']
print(classification_report(y_class,predictions_logistic_regression, target_names=target_names))


clf = VotingClassifier(estimators=[('bnb', MultinomialNB()), ('dt', svm.SVC(kernel='rbf', gamma=0.58, C=0.81, class_weight='balanced')), ('lr' , LogisticRegression(class_weight='balanced'))], n_jobs=1, voting='hard', weights=None)
predictions_voting_classifier = model.cross_val_predict(clf, X, y_class, cv=10)
print("****************************Voting Classifier*************************")
vc_avg_accuracy = class_accuracy(y_class, predictions_voting_classifier)
final_accuracies.append(vc_avg_accuracy)
target_names = ['Negative','Neutral','positive']
print(classification_report(y_class,predictions_voting_classifier, target_names=target_names))

clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,n_estimators=10, n_jobs=10, oob_score=False, random_state=None,
            verbose=0, warm_start=False)
predictions_randomForest_classifier = model.cross_val_predict(clf, X, y_class, cv=10)
print("****************************Random Forest Classifier*************************")
rf_avg_accuracy = class_accuracy(y_class, predictions_randomForest_classifier)
final_accuracies.append(rf_avg_accuracy)
target_names = ['Negative','Neutral','positive']
print(classification_report(y_class,predictions_randomForest_classifier, target_names=target_names))



data_frame_upsampled['nb_pred_class'] = predictions
data_frame_upsampled['svm_pred_class'] = predictions_svm
data_frame_upsampled['dtree_pred_class'] = predictions_decision_tree
data_frame_upsampled['lreg_pred_class'] = predictions_logistic_regression
data_frame_upsampled['vm_pred_class'] = predictions_voting_classifier
data_frame_upsampled['rf_pred_class'] = predictions_randomForest_classifier
data_frame_upsampled.to_excel("C:\\Users\\nikhi\\PycharmProjects\\Obama\\output_obama.xlsx")


# Plotting
df_plot = pd.DataFrame({'classifiers': ['Naive Bayes', 'SVM', 'Decision Trees', 'Logistic Regression', 'Voting Classifier', 'RandomForestClassifier'], 'values': final_accuracies})
ax = df_plot.plot.bar(x='classifiers', y='values', rot=0, width=0.30)
plot.show()






















































