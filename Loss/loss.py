#Loss functions in deepLearning
import numpy as np
#Mean Squared Error
def mse_loss(ytrue,ypred):
    return ((ytrue-ypred)**2).mean()

#Mean Absolute Error
def mae_loss(ytrue,ypred):
    return abs(ytrue-ypred).mean()