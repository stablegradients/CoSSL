import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

def get_metrics(outputs, labels, classes):
    '''
    returns a dictionary of computed metrics
    ARGS
        outputs: (np.ndarray) a (N, # classes) dimensional array of output logits of the model
        labels: (np.ndarray) a (N) dimensional array where each element is the ground truth
                index of the corresponding output element
        classes: (list) a list of stings of names of classes
    RETURNS:
        a dictionary of classification metircs, support for:
        1. precision,
        2. recall,
        3. accuracy,
        4. max precision across all classes
        5. mean precision across all classes
        6. min precision  across all classes
        7. max recall  across all classes
        8. mean recall across all classes
        9. min recall  across all classes
        10. f1 micro average
        11. f1 macroa average
        12. Head recall
        13. Tail recall
        14. Head Coverage
        15. Tail Coverage
    '''
    num_classes = len(classes)
    precision = precision_score(labels, outputs, average=None, zero_division=0)
    precision_avg = precision_score(labels, outputs, average='macro', zero_division=0)
    max_precision = np.max(precision)
    min_precision = np.min(precision)
    mean_precision = np.mean(precision)
    
    recall = recall_score(labels, outputs, average=None, zero_division=0)
    tail_recall = np.mean(recall[int(0.9*num_classes):])
    head_recall = np.mean(recall[:int(0.9*num_classes)])

    minHT = min(tail_recall, head_recall)

    recall_avg = recall_score(labels, outputs, average='macro', zero_division=0)
    max_recall = np.max(recall)
    min_recall = np.min(recall)
    mean_recall = np.mean(recall)

    f1_micro = f1_score(labels, outputs, average='micro')
    f1_macro = f1_score(labels, outputs, average='macro')
    
    CM = confusion_matrix(labels, outputs, normalize="all")
    coverages = np.sum(CM, axis=0)

    head_coverage, tail_coverage =  np.mean(coverages[:int(0.9*num_classes)]), \
                                    np.mean(coverages[int(0.9*num_classes):])
    accuracy = accuracy_score(labels, outputs)
    metrics =   {
                "precision": precision_avg,
                "recall": recall_avg,
                "accuracy": accuracy,
                "max_precision": max_precision,
                "mean_precision": mean_precision,
                "min_precision": min_precision,
                "max_recall": max_recall,
                "mean_recall": mean_recall,
                "min_recall": min_recall,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "tail_recall": tail_recall,
                "head_recall": head_recall,
                "min_head_tail": minHT,
                "head_coverage": head_coverage,
                "tail_coverage": tail_coverage
                }
    for i, name in enumerate(classes):
        metrics["precision_" + name] = precision[i]
        metrics["recall_" + name] = recall[i]

    for i, name in enumerate(classes):
        metrics["coverage_" + name] = coverages[i]
    
    metrics["min_coverage"] = min(coverages)
    return metrics, CM
