import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_textcnn_model(W, net, train_loader, epoch=5, lr=0.01):
    print("begin training")
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch): 
        total = 0.0
        correct = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()  
            data = data.tolist()
            data_vec = []
            for i in data:
                a = []
                for j in i:
                    a.append(W[j])
                data_vec.append(a)                          #id matrix convert to glove matrix
            data_vec = torch.Tensor(data_vec)
            data_vec = data_vec.unsqueeze(1)
            output = net(data_vec)  
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, target)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            loss.backward()
            optimizer.step()
        print('Accuracy of the network on train set: %f %%' % (100 * correct / total))
    print("end training")

def textcnn_model_test(W, net, test_loader):
    net.eval()  
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.tolist()
            data_vec = []
            for i in data:
                a = []
                for j in i:
                    a.append(W[j])
                data_vec.append(a)
            data_vec=torch.Tensor(data_vec)
            data_vec = data_vec.unsqueeze(1)
            output = net(data_vec )  
            _, predicted = torch.max(output.data, 1)  
            total += label.size(0)
            correct += (predicted == label).sum().item()
            print(predicted,label)
    print('Accuracy of the network on test set: %f %%' % (100 * correct / total))
    return total, correct
