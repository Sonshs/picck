import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from simple_kie.dataset import ShkDataSet
import numpy as np
import os
import cv2

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

def run_test(batch_size, data_path, model_path=""):
    ### init or loadmodel
    model = KieModel()
    weight = torch.load(model_path)

    params = model.parameters()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    print(optimizer)
    
    model.load_state_dict(weight["model_state_dict"], strict=False)
    model = model.to(device)
    CE_criterion = torch.nn.CrossEntropyLoss()

    ### prepare train/val dataset
    val_dataset = ShkDataSet(data_path=data_path, phase='test')
    val_num = val_dataset.__len__()
    print('Validation set size:', val_num)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True,
                                               pin_memory=True)

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
            
            outputs = model(samples) #.cuda())
            
            targets = targets.clone().detach()
            targets = targets.to(device) #.cuda()

            CE_loss = CE_criterion(outputs, targets)
            loss = CE_loss

            val_loss += loss
            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)
            correct_or_not = torch.eq(predicts, targets)
            print("target:")
            print(targets.cpu().tolist())
            bingo_cnt += correct_or_not.sum().cpu()
            
        val_loss = val_loss/iter_cnt
        val_acc = bingo_cnt.float()/float(val_num)
        val_acc = np.around(val_acc.numpy(), 4)
        print("Test accuracy:%.4f. Loss:%.3f" % (val_acc, val_loss))

def load_mode(model_path):
    ### init or loadmodel
    model = KieModel()
    weight = torch.load(model_path)

    params = model.parameters()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    print(optimizer)
    
    model.load_state_dict(weight["model_state_dict"], strict=False)
    model = model.to(device)
    CE_criterion = torch.nn.CrossEntropyLoss()

    return model

def inference(model, img_path, bbox_n_transcripts):
    img = img_path
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    hi, wi, _ = img.shape
    samples = []
    paragraph = []
    for box_n_transcript in bbox_n_transcripts:
        data = box_n_transcript.split(',')
        bbox = list(map(int,data[:8]))
        words = ','.join(data[8:])
        topleft = bbox[0:2]
        rightbottom = bbox[4:6]
        x,y = [(topleft[i]+rightbottom[i])/2 for i in range(2)]
        w,h = [rightbottom[i] - topleft[i] for i in range(2)]
        x,y,w,h = x/wi, y/hi, w/wi, h/hi
        word_count = len(words.split(' '))
        char_count = len(words)
        samples.append([x, y, w, h, word_count, char_count])
        paragraph.append(words)
    samples = torch.Tensor(samples)
    samples = samples.type(torch.cuda.FloatTensor).to(device)
    return model(samples), paragraph
    

    
epochs = 100
batch_size = 1
data_path = 'data/shk/only_points_n_rules'
output_path = 'saved/kie_model'

if __name__ == '__main__':
    model = load_mode(model_path="saved/kie_model/final_epoch50_acc0.7157_2.pth")
    bbox_n_transcripts = [
        "83,31,184,31,184,50,83,50,QUAN H??? V???I CH??? H???",
        "55,59,139,59,139,75,55,75,t??n g???i kh??c (n???u c??)",
        "32,71,118,71,118,89,32,89,Ng??y, th??ng, n??m sinh",
        "31,86,72,86,72,98,31,98,Qu?? qu??n",
        "211,82,240,82,240,95,211,95,Gi???i t??nh",
        "245,80,270,80,270,96,245,96,N???",
        "31,100,65,100,65,114,31,114,D??n t???c",
        "230,93,271,93,271,112,230,112,H?? N???i",
        "201,91,224,91,224,104,201,104,Oai",
        "148,104,188,104,188,118,148,118,Qu???c t???ch",
        "197,104,261,104,261,123,197,123,Vi???t Nam",
        "30,113,70,113,70,128,30,128,CMND s???",
        "168,119,218,119,218,132,168,132,H??? chi???u s???",
        "29,129,131,129,131,144,29,144,Ngh??? nghi???p, n??i l??m vi???c",
        "28,158,162,158,162,173,28,173,N??i th?????ng tr?? tr?????c khi chuy???n ?????n",
        "164,152,236,152,236,175,164,175,?????i s??? NK3a",
        "39,167,188,167,188,188,39,188,??MT ng??y 29/10/2010",
        "30,187,98,187,98,198,30,198,C??N B??? ????NG K??",
        "24,303,93,303,93,315,24,315,C??N B??? ????NG K??",
        "31,197,94,197,94,208,31,208,(K??, ghi r?? h??? t??n)",
        "26,314,89,314,89,323,26,323,(K??, ghi r?? h??? t??n)",
        "22,274,136,274,136,289,22,289,L?? do x??a ????ng k?? th?????ng tr??",
        "124,302,150,302,150,314,124,314,Ng??y",
        "163,301,190,301,190,312,163,312,th??ng",
        "207,300,225,300,225,310,207,310,n??m",
        "116,313,183,313,183,326,116,326,TR?????NG C??NG AN",
        "19,244,118,244,118,269,19,269,Tr???nh Th??? H??",
        "146,253,274,253,274,273,146,273,Th?????ng t?? ????o Huy S???i",
        "120,200,185,200,185,212,120,212,TR?????NG C??NG AN",
        "128,190,152,190,152,200,128,200,Ng??y",
        "167,189,192,189,192,201,167,201,th??ng",
        "206,192,225,192,225,201,206,201,n??m",
        "224,187,249,187,249,201,224,201,2014",
        "154,186,172,186,172,199,154,199,26",
        "193,187,206,187,206,202,193,202,9",
        "142,240,279,240,279,254,142,254,P.TR?????NG C??NG AN HUY???N",
        "146,320,240,320,240,334,146,334,(K??, ghi r?? h??? t??n v?? ????ng d???u)"
    ]

    img = "data/shk/images/000001_1_ch.jpg"
    # results, paragraph = inference(model, img, bbox_n_transcripts)
    # _, predicts = torch.max(results, 1)
    # for i, predict in enumerate(predicts):
    #     print(predict.item(), ' - ', paragraph[i])

    run_test(8,"data/shk/only_points_n_rules", "saved/kie_model/final_epoch50_acc0.7157_2.pth")
