from torch.optim import Adam
from tqdm import tqdm
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
import torch

def train_Model(model,Train_DL, learning_rate, epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Loss = BCELoss()
    Optimizer = Adam(params=model.parameters(), lr=learning_rate)
    scheduler = StepLR(Optimizer, step_size=212, gamma=0.1)

    model.to(device)
    model.train()

    train_acc_epochs = []
    train_loss_epochs = []
    val_acc_epochs = []
    val_loss_epochs = []

    for epoch in range(epochs):
        training_loss = {}
        training_accuracy = {}
        validation_loss = {}
        validation_accuracy = {}
        batch = 0

        for comments, labels in tqdm(Train_DL):

            labels = labels.to(device)
            labels = labels.float()
            masks = comments["attention_mask"].squeeze(1).to(device) # the model used these masks to attend only to the non-padded tokens in the sequence
            input_ids = comments["input_ids"].squeeze(1).to(device) # contains the tokenized and indexed representation for a batch of comments
            # squeeze is used to remove the second dimension which has size 1.
            output = model(input_ids, masks) # vector of logits for each class
            loss = Loss(output.logits, labels) # compute the loss

            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
            scheduler.step()


            batch += 1
            if batch%53 == 0:
                with torch.no_grad():
                    acc = []
                    op = output.logits
                    for lb in range(len(labels)): # note: labels is of shape (batch_size, num_classes(=7))
                        correct = 0
                        for i in range(len(labels[lb])):  # therefore len(labels[lb]) is 7
                            res = 1 if op[lb,i]>0.5 else 0
                            if res == labels[lb,i]:
                                correct += 1
                        acc.append(correct/len(labels[lb]))

                    training_loss[batch] = loss.item()
                    training_accuracy[batch] = sum(acc)/len(acc)
                    print(f"Epoch:{epoch+1} | batch no:{batch}/{len(Train_DL)} | Loss:{loss.item():.4f} | Accuracy:{sum(acc)/len(acc):.4f}")



        train_acc_epochs.append(training_accuracy)
        train_loss_epochs.append(training_loss)
        

    return train_acc_epochs, train_loss_epochs, val_acc_epochs, val_loss_epochs