
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
import torch

def evaluate_Model(model,Test_DL):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Loss = BCELoss()
    model.to(device)
    model.eval()

    accTest = []
    Test_loss = 0
    for comments, labels in Test_DL:
        labels = labels.to(device)
        labels = labels.float()
        masks = comments["attention_mask"].squeeze(1).to(device)
        input_ids = comments["input_ids"].squeeze(1).to(device)

        output = model(input_ids, masks)
        loss = Loss(output.logits, labels)
        Test_loss += loss.item()

        op = output.logits
        correct_val = 0
        for i in range(7):
            res = 1 if op[0,i]>0.5 else 0
            if res == labels[0,i]:
                correct_val += 1
        accTest.append(correct_val/7)

    print("Testing Dataset:\n")
    print(f" Test Loss:{Test_loss/len(Test_DL):.4f} | Test Accuracy:{sum(accTest)/len(accTest):.4f}")