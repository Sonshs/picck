import torch
import torch.utils.data as data
import os
import cv2

class ShkDataSet(data.Dataset):
    def __init__(self, data_path, phase=""):
        # train_file = os.path.join(data_path, 'train_samples_list.csv')
        # self.index_labels = ["adult","child"]
        # self.phase = phase
        data_folder = os.path.join(data_path, phase)

        files = [f for f in os.listdir(data_folder)]
        self.samples = []
        self.labels = []
        for file in files:
            file_name = os.path.join(data_folder, file)
            with open(file_name, 'r') as fr:
                for row in fr:
                    data = row[:-1].split(',')
                    xywh = list(map(float, data[1:5]))
                    word_count = int(data[-3])
                    char_count = int(data[-2])
                    try:
                        label = int(data[-1])
                    except:
                        print(file_name)
                        print(data[0], data[-1])
                    sample = xywh + [word_count, char_count]
                    self.samples.append(sample)
                    self.labels.append(label)
        self.samples, self.labels = zip(*sorted(zip(self.samples, self.labels)))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample = torch.Tensor(sample)
        # print("sample: ", sample)
        label_idx = self.labels[idx]
        # label = [0]*39
        # label[label_idx-1] = 1
        # print("type dataset img: {}".format(img.type()))
        return sample, label_idx-1, idx