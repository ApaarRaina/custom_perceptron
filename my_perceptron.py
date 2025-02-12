from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt


input=4
output=1
X,y=make_classification(n_samples=300,n_features=input,n_classes=2,n_clusters_per_class=1,random_state=42)


X=np.array(X)
y=np.array(y)



def binaryloss(y_pred,y_true):
    epsilon = 1e-9  # Small value to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))



def compute_gradient(lr,X,y_pred,y_true,model):
    X_gradient=X
    X_gradient=np.dot(X_gradient.T,(y_pred-y_true))/len(y_true)
    model.weights=model.weights-lr*X_gradient

    return X_gradient


class perceptron():

    def __init__(self,input,output):
        self.weights = np.random.rand(input,output)


    def forward(self,X):

        output=np.matmul(X,self.weights)

        return output


model=perceptron(input,output)

epochs=1000
losses=[]
lr=0.01
threshold=1e-5

for i in range(epochs):

    output=model.forward(X)
    y_pred=1/(1+np.exp(-output))

    loss=binaryloss(y_pred,y)
    losses.append(loss)
    print(f"The loss is {loss}")

    gradient=compute_gradient(lr,X,y_pred,y,model)



plt.figure(figsize=(5,5))
plt.plot(range(epochs),losses)
plt.show()



