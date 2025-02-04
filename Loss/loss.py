#Loss functions in deepLearning
import numpy as np
#Mean Squared Error
def mse_loss(ytrue,ypred):
    return ((ytrue-ypred)**2).mean()

#Mean Absolute Error
def mae_loss(ytrue,ypred):
    return abs(ytrue-ypred).mean()

#Mean Squared Logarithmic Error
def msle(ytrue,ypred):
    ytrue = np.array(ytrue)
    ypred = np.array(ypred)

    ytrue = np.log(ytrue+1)
    ypred = np.log(ypred+1)

    squared = np.square(ytrue - ypred)
    msle_value = np.mean(squared)
    return msle_value

#Binary CrossEntropy
def binary_crossentropy(ytrue,ypred,epsilon=1e-15):
    ypred = np.clip(ypred,epsilon,1-epsilon)
    return -1*np.mean((ytrue*np.log(ypred))+((1-ytrue)*np.log(1-ypred)))

##yet to be updated