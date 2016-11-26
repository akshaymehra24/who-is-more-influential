import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from sklearn import linear_model
# from sklearn.metrics import roc_auc_score

###########################
# LOADING TRAINING DATA
###########################

DATA_DIR = 'C:/Users/Patrick/PycharmProjects/who-is-more-influential/Data Analysis'
OUTPUT_DIR = 'C:/Users/Patrick/PycharmProjects/who-is-more-influential/Python'
SAVE_DIR = 'C:/Users/Patrick/PycharmProjects/who-is-more-influential/Python/Plots'

# (filename, range_x, range_y, title/output)
data_sets = [
    ('train_normalized_delta.csv', (1, 12), (0, 1), 'Logistic Regression - Normalized Difference'),
    ('train_new_discrete.csv', (2, 13), (1, 2), 'Logistic Regression - Binary Transform'),
    ('train_log.csv', (1, 12), (0, 1), 'Logistic Regression - Log Transform'),
]

# Run & Plot each data set
for data_set in data_sets:
    filename, x_range, y_range, title = data_set
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
    y_train = np.array(y)
    X = np.array(x)
    x_train = X[:, [0, 1]]

    # Shuffle and split training and test sets
    model = linear_model.LogisticRegression(fit_intercept=False)
    model.fit(x_train, y_train)

    h = .02  # step size in the mesh
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for j, model in enumerate([model]):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, j + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
        plt.xlabel('Followers')
        plt.ylabel('Following')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(title)

    # Output
    # output = join(SAVE_DIR, title)
    # plt.savefig(output)
    plt.show()
