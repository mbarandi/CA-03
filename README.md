# CA-03
The purpose of this program is to build a decision tree model to predict the income level of individuals. To achieve this, the program uses a training dataset to train the model and a separate test dataset to evaluate its accuracy. By using specific parameters, the model can predict the income level of a subject, and the program reports the accuracy of this prediction.

To complete the project, the following imports are necessary:
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib notebook
%matplotlib inline
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from six import StringIO
from IPython.display import Image  
import graphviz
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
