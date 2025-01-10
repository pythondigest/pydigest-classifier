"""
Plots confusion matrix and prints some metrics computed on input report file (command line arg).
Report file is a csv file with named columns. Names are: ["link", "classificator", "moderator"], order is arbitrary.

```
link,classificator,moderator
http://habrahabr.ru/post/200376/,False,False
.....
"""

import csv
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def to_bool(s: str) -> bool:
    return s == "True"


input_path = sys.argv[1]  # report.csv
with open(input_path) as csvreport:
    reportreader = csv.DictReader(csvreport, delimiter=",")
    y_true = []
    y_pred = []
    for row in reportreader:
        y_true.append(to_bool(row["moderator"]))
        y_pred.append(to_bool(row["classificator"]))
print(len(y_true), sum(y_true))

conf = confusion_matrix(y_true, y_pred, labels=[True, False])
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 score:", f1_score(y_true, y_pred))
sns.heatmap(
    conf,
    fmt="d",
    cbar=False,
    xticklabels=["Predicted: True", "Predicted: False"],
    yticklabels=["Real: True", "Real: False"],
)
plt.show()
