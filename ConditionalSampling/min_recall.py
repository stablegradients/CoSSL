import numpy as np
import torch
import torch.nn.functional as F


def ulb_sample_distribution(CM, lambdas):
    num_classes = CM.shape[0]
    CM = CM/np.sum(CM)  # normalising the distribution
    CM_row_sum = np.sum(CM, axis=1)  # a commonly used term in the derivative 
    derivative_matrix = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(num_classes):
            if j==i:
                derivative_matrix[i,j] = (-1 * lambdas[i]/CM_row_sum[i]) + (-1 * lambdas[i] * CM[i,i]/(CM_row_sum[i]**2))
            else:
                derivative_matrix[i,j] = -1 * lambdas[i]/(CM_row_sum[i]**2)

    negative_derivative_matrix = torch.tensor(-1 * derivative_matrix)
    return F.softmax(negative_derivative_matrix, dim=1)
        
def lambda_update(recall_vector, lambdas, val_lr):
    new_lamdas = [x * np.exp(-1 * val_lr * r) for x, r in zip(lambdas, recall_vector.tolist())]
    new_lamdas = [x/sum(new_lamdas) for x in new_lamdas]
    return new_lamdas