import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def precision_and_recall(ALLFILES, queryimg_index, dst, labels):
    precision = []
    recall = []
    labels_by_rank = np.array([], dtype = int)
    label_query = ALLFILES[queryimg_index][1]
    for i in range(len(ALLFILES)):
        labels_by_rank = np.append(labels_by_rank, ALLFILES[dst[i][1]][1])
        no_rank_same_query = np.sum((labels_by_rank == label_query).astype(int))
        precision.append(no_rank_same_query / len(labels_by_rank))
        no_label_query_data = labels[label_query]

        recall.append(no_rank_same_query / no_label_query_data )

    labels_by_rank = labels_by_rank.astype(int)

    return precision, recall, labels_by_rank

def plotPRcurve(precision, recall, SHOW):
    plt.plot(recall[0:SHOW], precision[0:SHOW], linestyle='-', color='r')#
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.show()

def confusionMatrix(ALLFILES, queryimg_index, dst, SHOW, labels_by_rank):
    confusion_matrix = np.zeros((20, 20), dtype = int)
    label_query = int(ALLFILES[queryimg_index][1]) if ALLFILES[queryimg_index][1] is not None else None
    
    for i in range(SHOW):
        confusion_matrix[label_query - 1, labels_by_rank[i] - 1] += 1
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)

    ax.set_xticks(np.arange(20) + 0.5, minor=False)
    ax.set_yticks(np.arange(20) + 0.5, minor=False)

    ax.set_xticklabels(np.arange(1, 21), rotation=0)
    ax.set_yticklabels(np.arange(1, 21), rotation=0)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()