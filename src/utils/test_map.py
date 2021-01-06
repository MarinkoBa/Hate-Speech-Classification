import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pandas as pd

#TODO add docstring


def test_map(y_pred,y_test):
    cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
    
    
    class_names=[0,1]
    fig, ax = plt.subplots()
    tick_marks=np.arange(len(class_names))
    plt.xticks(tick_marks,class_names)
    plt.yticks(tick_marks,class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu",fmt="g")
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title("Matrix",y=1.1)
    plt.ylabel("actual label")
    plt.xlabel("predicted label")
    plt.savefig("mygraph.png")
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    
    


