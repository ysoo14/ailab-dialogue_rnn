import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
from transformers import *
from sklearn.manifold import TSNE
import numpy as np

from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class IEMOCAPDataset(Dataset):

    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)
        self.csv_file = './IEMOCAP_features/scriptDataset1.csv'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encodingModel = BertModel.from_pretrained('bert-base-uncased').to(self.device)

        for param in self.encodingModel.parameters():
            param.requires_grad = False

    def __getitem__(self, index):
        vid = self.keys[index]
        ex =  self.bertEncoding(vid)

        return torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               ex,\
               vid

    def __len__(self):
        return self.len

    def bertEncoding(self, vid):
        rSentences = self.getSentences(vid)
        inputs = self.tokenizer(rSentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        inputs['input_ids'] = inputs['input_ids'].view(-1, len(rSentences))
        inputs['token_type_ids'] = inputs['token_type_ids'].view(-1, len(rSentences))
        inputs['attention_mask'] = inputs['attention_mask'].view(-1, len(rSentences))
        outputs = self.encodingModel(**inputs)
        encoded = outputs.last_hidden_state[0]
        del inputs
        del outputs
        return encoded

    def getSentences(self, vid):
        rSentences = []
        utteranceId = self.videoIDs[vid]
        with open(self.csv_file, 'r') as csvfile:
            sentences = csvfile.readlines()
            df = pd.read_csv(self.csv_file)
    
            for u_id in utteranceId:
                rSentences.append(df[df.fileName == u_id].script.item())

        return rSentences

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<1 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]


class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence,\
            self.trainVid, self.testVid = pickle.load(open(path, 'rb'),encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path, classify, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabelsEmotion, self.videoText,\
        self.videoAudio, self.videoSentence, self.trainVid,\
        self.testVid, self.videoLabelsSentiment = pickle.load(open(path, 'rb'))

        if classify == 'emotion':
            self.videoLabels = self.videoLabelsEmotion
        else:
            self.videoLabels = self.videoLabelsSentiment
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):
        
        self.Speakers, self.Features, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return  torch.FloatTensor(self.Features[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.EmotionLabels[conv])), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                conv

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]


class DailyDialogueDataset2(Dataset):

    def __init__(self, split, path):

        self.Speakers, self.Features, _, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]

        return torch.FloatTensor(self.Features[conv]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.Speakers[conv]]), \
               torch.FloatTensor([1] * len(self.EmotionLabels[conv])), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               conv

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]

# def tsne_visualize(feature, label, fileName):
#     colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
#                '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']

#     tsne = TSNE(n_components=2, verbose=1, n_iter=300, perplexity=5)
#     tsne_v = tsne.fit_transform(feature)

#     for i in range(len(feature)): 
#         plt.text(tsne_v[i, 0], tsne_v[i, 1], label[int(feature['labels'][i])], # x, y, 그룹; str은 문자로 변환
#                 color=colors[int(feature['labels'][i])], # 산점도 색상
#                 fontdict={'weight':'bold', 'size':9}) # font 설정

#     plt.xlim(tsne_v[:, 0].min(), tsne_v[:,0].max()) # 최소, 최대
#     plt.ylim(tsne_v[:, 1].min(), tsne_v[:,1].max()) # 최소, 최대
#     plt.xlabel('first principle component') # x 축 이름
#     plt.ylabel('second principle componet') # y 축 이름
#     plt.savefig(str(fileName) + '.png', dpi=300)

# if __name__ == '__main__':

#     data = IEMOCAPDataset()
    
#     cnn = np.array([])
#     bert = np.array([])
#     label = np.array([])
#     idx = 0

#     for d in data:
#         label = np.concatenate((label,d[3].cpu().detach().numpy()))
#         if idx == 0:
#             cnn = d[0].cpu().detach().numpy()
#             bert = d[4].cpu().detach().numpy()
#         else:
#             cnn = np.vstack((cnn,d[0].cpu().detach().numpy()))
#             bert = np.vstack((bert, d[4].cpu().detach().numpy()))

#         idx += 1
    
#     labels = pd.DataFrame(label)
#     labels.columns = ['labels']
#     data_cnn = pd.DataFrame(cnn)
#     feature_cnn = pd.concat([data_cnn,labels],axis=1)

#     data_bert = pd.DataFrame(bert)
#     feature_bert = pd.concat([data_cnn,labels],axis=1)

#     label_str = ['h', 's', 'n', 'a', 'e', 'f']
#     tsne_visualize(feature=feature_bert, label=label_str, fileName='bert')
#     tsne_visualize(feature=feature_cnn, label=label_str, fileName='cnn')
