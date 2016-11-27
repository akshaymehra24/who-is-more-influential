import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from os.path import join
from sklearn import linear_model

###########################
# LOADING TRAINING DATA
###########################

DATA_DIR = 'C:/Users/Patrick/PycharmProjects/who-is-more-influential/Data Analysis'
OUTPUT_DIR = 'C:/Users/Patrick/PycharmProjects/who-is-more-influential/Python'
SAVE_DIR = 'C:/Users/Patrick/PycharmProjects/who-is-more-influential/Python/Plots'

# (filename, range_x, range_y, plot_color, plot_label)
data_set = ('train_log.csv', (1, 12), (0, 1))

exclude_features = [
    # Index of feature in feature list (not data index), plot_color, plot_label
    ([], 'b-', 'All                 '),
    # ([3, 7, 10], 'r-', 'NW3, Posts, MRec'),     # Mentions Received, Posts, Network 3
    ([1], 'g-', '- Following    '),  # + Following Count
    ([3], 'r-', '- Ment. Rec.  '),  # + Following Count
    ([1, 3], 'c-', '- (Both)         '),  # + Following Count
]
# Load data
filename, x_range, y_range = data_set
trainfile = open(join(DATA_DIR, filename))
header = trainfile.next().rstrip().split(',')

x = []
y = []

# Read data from the training file
for line in trainfile:
    columns = line.rstrip().split(',')
    label = int(columns[y_range[0]:y_range[1]][0])
    features = [float(item) for item in columns[x_range[0]:x_range[1]]]
    x.append(features)
    y.append(label)
trainfile.close()

# Process each subset of features
for fconfig in exclude_features:
    fset, plot_color, plot_label = fconfig
    use_features = [item for i, item in enumerate(x) if i not in fset]
    use_labels = [item for i, item in enumerate(y) if i not in fset]
    # Convery to numpy array
    X = np.array(use_features)
    Y = np.array(use_labels)

    # Shuffle and split training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.5, random_state=0)

    # Run the model
    model = linear_model.LogisticRegression(fit_intercept=False)
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_test)[:, 1]

    # Analyze the results
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_label = '{0} {1:0.5f}'.format(plot_label, roc_auc)

    # Graph results
    plt.plot(false_positive_rate, true_positive_rate, plot_color, label=roc_label, linewidth=2)

# Graph Labels
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')     # plot the diagonal
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# Output
output = join(SAVE_DIR, 'Logistic Regression ROC - Feature Selection')
plt.savefig(output)
# plt.show()
