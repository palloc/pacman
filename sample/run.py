from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

from sample.utils import data_loader


XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_1.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'
STOP_WORDS = ['']


def run():
    """
    データ作成
    """
    xss_train_data, xss_train_label = data_loader(XSS_TRAIN_FILE, 'xss')
    xss_test_data, xss_test_label = data_loader(XSS_TEST_FILE, 'xss')
    normal_train_data, normal_train_label = data_loader(NORMAL_TRAIN_FILE, 'normal')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE, 'normal')

    X_train = xss_train_data + normal_train_data
    y_train = xss_train_label + normal_train_label
    X_test = xss_test_data + normal_test_data
    y_test = xss_test_label + normal_test_label


    """
    データ前処理・学習機作成
    """
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w\w+\b|/|\)|;', stop_words=STOP_WORDS)
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)


    """
    テスト
    """
    X_test = vectorizer.transform(X_test)
    pred = clf.predict(X_test)
    acc_score = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(
        pred, y_test, labels=['xss', 'normal']
    )
    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)
        

if __name__ == '__main__':
    run()
