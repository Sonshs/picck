import os
import json


t = 0
f_pred = 0
f_not_detect = 0

gt_folder = 'data/shk/boxes_and_transcripts'
predict_folder = 'saved/test_output/model_shk'

# gts = [f for f in os.listdir(gt_folder)] # mix train-test
file_names = [f[:-5] for f in os.listdir(predict_folder)] # json ext -> [:-5]

file_names.sort()


for idx in range(len(file_names)):
    gt = os.path.join(gt_folder,file_names[idx] + '.tsv')
    pred = os.path.join(predict_folder, file_names[idx] + '.json')
    d_gt = {}
    with open(gt, 'r') as gt:
        for row in gt:
            data = row[:-1].split(',')
            value = ','.join(data[9:-1])
            label = data[-1]
            if label != '30':
                d_gt[value] = [label]
    d_pred = {}
    with open(pred, 'r') as pred:
        data = json.load(pred)
        for ele in data:
            d_pred[data[ele]] = ele

    for ele in d_gt:
        if ele in d_pred:
            if d_pred[ele] in d_gt[ele]:
                t += 1
            else:
                f_pred += 1
        else:
            f_not_detect += 1
    
print("true: ", t)
print("total false: ",f_not_detect+f_pred)
print(" -- false pred: ", f_pred)
print(" -- false npt detect: ",f_not_detect)
