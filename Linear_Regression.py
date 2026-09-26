import torch
from torch import nn
import matplotlib.pyplot as plt
#Setting up device
device = "cuda" if torch.cuda.is_available() else "cpu"

#Create data
weight = 0.4
b = 0.5

X = torch.arange(0, 1, 0.01).unsqueeze(1)
y = weight * X + b

#Split data
spliter = int(0.8 * len(X))

X_train, y_train = X[:spliter], y[:spliter]
X_test, y_test = X[spliter:], y[spliter:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features= 1, out_features= 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
torch.manual_seed(0)

def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test,
                    test_labels=y_test, prediction=None):
    plt.figure()
    plt.scatter(train_data, train_labels, c='r', s= 10, label= "Train data")
    plt.scatter(test_data, test_labels, c='g', s= 10, label= "Test data" )

    if prediction is not None:
        plt.scatter(test_data, prediction, c='b', s= 10, label= "Prediction")

    plt.legend()
    plt.show()

plot_predictions()


model = LinearRegression()
model.to(device)
print(next(model.parameters()).device)

X_train, X_test, y_train, y_test = map(lambda x: x.to(device),(X_train, X_test, y_train, y_test))

# Create loss function
loss_fn = nn.L1Loss()

# Create an optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

# Set the number of epochs
epochs = 1000

for epoch in range(epochs):
    #Train
    model.train()
    # 1. Forward pass
    y_pred = model(X_train)
    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)
    # 3. Zero gradient optimizer
    optimizer.zero_grad()
    # 4. Backpropagation
    loss.backward()
    # 5. Gradient descent
    optimizer.step()

    #Evaluate
    model.eval()

    with torch.inference_mode():
        test_pred = model(X_test)

        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

with torch.inference_mode():
    y_preds = model(X_test)

plot_predictions(prediction= y_preds.cpu())



