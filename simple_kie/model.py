import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from simple_kie.dataset import ShkDataSet
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KieModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6,40)
        self.fc2 = nn.Linear(40,80)
        self.fc3 = nn.Linear(80,39)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def run_training(epochs, batch_size, data_path, output_path, model_path=""):
    ### init or loadmodel
    if model_path:
        model = KieModel()
        weight = torch.load(model_path)
        model.load_state_dict(weight["model_state_dict"], strict=False)
        # model = model.to(device)
    else:
        model = KieModel()
        model.apply(init_weights)

    params = model.parameters()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    print(optimizer)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    model = model.to(device) #.cuda()
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    # weights = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    # class_weights = torch.FloatTensor(weights).cuda()
    # self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # CE_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    CE_criterion = torch.nn.CrossEntropyLoss()

    ### prepare train/val dataset
    train_dataset = ShkDataSet(data_path=data_path, phase='train')
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True,
                                               pin_memory=True)

    val_dataset = ShkDataSet(data_path=data_path, phase='test')
    val_num = val_dataset.__len__()
    print('Validation set size:', val_num)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True,
                                               pin_memory=True)

    ### start training
    best_acc = 0
    for i in range(1, epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        for batch_i, (samples, targets, _) in enumerate(train_loader):
            # print("imgs shape: {}".format(imgs.shape))
            # print("type of imgs: {}".format(imgs.type()))
            iter_cnt += 1
            optimizer.zero_grad()
            # imgs = imgs.to(device) #.cuda()
            # imgs = torch.Tensor(imgs).to(device)
            # imgs.div_(255).sub_(0.5).div_(0.5)
            # samples = torch.Tensor(samples).to(device)
            # samples = samples.type(torch.cuda.FloatTensor)
            # print(len(samples))
            # print(samples[0].shape)
            # print(targets.shape)
            samples = torch.Tensor(samples)
            samples = samples.type(torch.cuda.FloatTensor).to(device)
            outputs = model(samples)
            # outputs = []
            # for img in imgs:
            #     output= model(img)
            #     outputs.append(output)
            # print("targets: ")
            # print(targets)
            # targets = torch.tensor(targets)
            targets = targets.clone().detach()
            targets = targets.to(device) #.cuda()

            # CE_loss = CE_criterion(outputs, targets)
            # loss = CE_loss
            loss = CE_criterion(outputs, targets)
            loss.backward()

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
                # scaled_loss.backward()
            optimizer.step()
            
            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
                

        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f' %
              (i, train_acc, train_loss, optimizer.param_groups[0]["lr"]))
        scheduler.step()

        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for batch_i, (samples, targets, _) in enumerate(val_loader):
                # imgs = imgs.to(device)
                # imgs = torch.Tensor(imgs).to(device)
                # imgs.div_(255).sub_(0.5).div_(0.5)

                samples = torch.Tensor(samples)
                samples = samples.type(torch.cuda.FloatTensor).to(device)
                # samples = torch.Tensor(samples).to(device)
                # print("--------------------------")
                # print(samples.shape)
                outputs = model(samples) #.cuda())
                # outputs = []
                # for img in imgs:
                #     output= model(img)
                #     outputs.append(output)
                # print(targets)
                # targets = torch.tensor(targets)
                targets = targets.clone().detach()
                targets = targets.to(device) #.cuda()
                # print('targets ')
                # print(targets.min(), targets.max())
                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                
            val_loss = val_loss/iter_cnt
            val_acc = bingo_cnt.float()/float(val_num)
            val_acc = np.around(val_acc.numpy(), 4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, val_acc, val_loss))

            if val_acc > best_acc:
                best_acc = val_acc
                print("best_acc:" + str(best_acc))
            #     best_model = copy.deepcopy(model)

            if i == epochs:
                # torch.save(best_model,
                #            os.path.join(output_path, "best_model" + str(i) + "_acc" + str(best_acc) + "_2.pth"))
                # print('best model saved.')

                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join(output_path, "final_epoch" + str(i) + "_acc" + str(val_acc) + "_2.pth"))
                print('Model final saved.')

epochs = 100
batch_size = 1
data_path = 'data/shk/only_points_n_rules'
output_path = 'saved/kie_model'


run_training(epochs, batch_size, data_path, output_path, model_path="")