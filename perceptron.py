from sklearn.datasets import make_classification
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


input=4
X,y=make_classification(n_samples=300,n_features=input,n_classes=2,n_clusters_per_class=1,random_state=42)


X=np.array(X)
y=np.array(y)


class perceptron(nn.Module):

    def __init__(self,input):
        super().__init__()
        self.fc1=nn.Linear(input,1)

    def forward(self,x):
        output=self.fc1(x)
        output=F.sigmoid(output)

        return output


model=perceptron(input)
criterion=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


model.train()

epochs=300

losses=[]
X=torch.tensor(X)
X=X.to(torch.float32)
y=torch.tensor(y)
y=y.to(torch.float32)

for i in range(300):
    y_pred=model(X)
    y_pred=y_pred.ravel()
    loss=criterion(y_pred,y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.figure(figsize=(5,5))
plt.plot(range(epochs),losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("The loss of a pytorch perceptron")
plt.show()





