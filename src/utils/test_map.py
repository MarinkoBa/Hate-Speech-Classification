import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pandas as pd

def test_map(y_pred,y_test):
    """
    Get accurracy, precision and recall and save png of true positive,
    false positive, true negative and false negative for given prediction and test data 

    Parameters
    ----------
    prediction:         Binary array
                        array containing preddiction for test data for a model
    y_test:		        Pandas dataframe
                      	The dataframe containing the test labels.             
    """
    
    # get confusion matrix containig predicted and true labels
    cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
    
    # set up plot
    class_names=[0,1]
    fig, ax = plt.subplots()
    tick_marks=np.arange(len(class_names))
    plt.xticks(tick_marks,class_names)
    plt.yticks(tick_marks,class_names)
    
    # set up heatmap and labels
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu",fmt="g")
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title("Matrix",y=1.1)
    plt.ylabel("actual label")
    plt.xlabel("predicted label")
    
    # save plot
    plt.savefig("mygraph.png")
    
    # print acurracy, precision and recall
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    
    


