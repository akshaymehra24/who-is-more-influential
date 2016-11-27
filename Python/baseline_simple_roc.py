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
data_sets = [
    ('train_new.csv', (2, 13), (1, 2), 'g-', 'Baseline'),
    # ('train_new_discrete.csv', (2, 13), (1, 2), 'c-', 'BIN '),
    # ('pca_train.csv', (1, 7), (7, 8), 'r-', 'PCA'),
    # ('train_log.csv', (1, 12), (0, 1), 'b-', 'LOG'),
]

# Run & Plot each data set
for data_set in data_sets:
    filename, x_range, y_range, plot_color, plot_label = data_set
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

    # Convery to numpy array
    Y = np.array(y)
    X = np.array(x)

    # Run the model
    y_pred = [(1 if x[0] > 0 else 0) for x in X]      # higher follower count

    # Analyze the results
    false_positive_rate, true_positive_rate, _ = roc_curve(y, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_label = '{0} {1:0.3f}'.format(plot_label, roc_auc)

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
output = join(SAVE_DIR, 'ROC - Baseline')
plt.savefig(output)
# plt.show()
