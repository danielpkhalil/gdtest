import torch
from torch.utils.data import DataLoader
import hydra
from hydra.utils import instantiate
import os

def train_step(classifier, model, optimizer, criterion, x, t, y, project_fn=None):
    '''
    classifier: the classifier model
    model: the discrete diffusion model
    '''
    optimizer.zero_grad()
    xt = model.q_sample(x, t) # sample from q(x_t|x_0)
    xt = project_fn(xt) if project_fn is not None else xt
    y_pred = classifier(xt, t)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_model(classifier, model, dataloader, n_epochs=100, project_fn=None):
    classifier = classifier.to(model.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = torch.nn.MSELoss()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x, y in dataloader:
            t = torch.randint(0, model.timestep, (x.shape[0],)).to(torch.long)
            x = x.to(classifier.device)
            y = y.to(classifier.device)
            t = t.to(classifier.device)
            loss = train_step(classifier, model, optimizer, criterion, x, t, y, project_fn)
            epoch_loss += loss
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader)}")
    return classifier
