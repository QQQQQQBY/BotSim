
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from sklearn.utils import shuffle
# from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import datetime
import logging
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv

import torch.nn.functional as F
from torch_geometric.data import Data




def get_logger(name,log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(filename=log_dir,mode='w+',encoding='utf-8')
    fileHandler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s|%(levelname)-8s|%(filename)10s:%(lineno)4s|%(message)s")
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger


def load_data(seed):
    data = Data()
    num = torch.load("BotSim-24-Exp/Dataset/num_properties_tensor.pt").cuda()
    text = torch.load("BotSim-24-Exp/Dataset/text_tensor.pt").cuda()
    feature = torch.cat((num, text), dim=1)
   
    data.edge_index = torch.load('BotSim-24-Exp/Dataset/edge_index.pt').T.long().cuda()
    data.edge_type = torch.load('BotSim-24-Exp/Dataset/edge_type.pt').long().cuda()
    data.edge_weight = torch.load('BotSim-24-Exp/Dataset/edge_weight.pt').float().cuda()
    # label
    label = torch.load('BotSim-24-Exp/Dataset/label.pt')
    data.x = feature

    data.y = label
    def sample_mask(idx, l):
        """Create mask."""
        mask = torch.zeros(l)
        mask[idx] = 1
        return torch.as_tensor(mask, dtype=torch.bool)
    sample_number = len(data.y)

    shuffled_idx = shuffle(np.array(range(len(data.y))), random_state=seed) # 已经被随机打乱

    train_idx = shuffled_idx[:int(0.7* data.y.shape[0])].tolist()
    val_idx = shuffled_idx[int(0.7*data.y.shape[0]): int(0.9*data.y.shape[0])].tolist()
    test_idx = shuffled_idx[int(0.9*data.y.shape[0]):].tolist()
    train_mask = sample_mask(train_idx, sample_number)
    val_mask = sample_mask(val_idx, sample_number)
    test_mask = sample_mask(test_idx, sample_number)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data,train_idx, val_idx, test_idx



class GCN(nn.Module):
    def __init__(self,value_num, text_num,feature_dim,classes,dropout,data,num_layers):
        super(GCN,self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(value_num,feature_dim)
        self.fc2 = nn.Linear(text_num,feature_dim)
        
        self.relu = nn.Sequential(
            nn.Linear(feature_dim * 2,feature_dim),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(0.3)
        )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.rgcn = GCNConv(feature_dim,feature_dim,dropout=0.3)
            self.convs.append(self.rgcn)
        self.fc3 = nn.Linear(feature_dim,classes)


   
    def forward(self,value_feature, text_feature,edge_index,edge_weight,idx):
        value_feature = self.fc1(value_feature)
        text_feature = self.fc2(text_feature)
        feature = torch.concat((value_feature, text_feature),dim=1)
        feature = self.relu(feature)          
        
        for conv in self.convs:
            feature = conv(feature, edge_index=edge_index, edge_weight = edge_weight)     
                    
        output = self.fc3(feature[idx])              
        return output
        

loss_ce = nn.CrossEntropyLoss()

def train(data, train_idx, val_idx, test_idx,model,num_epochs,lr,weight_decay,logger,model_file):
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    max_acc = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_acc_total = 0
        train_loss_total = 0
        data = data.cuda()
        label = data.y.cuda()  
        label = label[train_idx].long()
        value_feature = data.x[:,:10].cuda()
        text_feature = data.x[:, 10:].cuda()                  
        output = model(value_feature, text_feature,data.edge_index,data.edge_weight, train_idx).float()        
        loss = loss_ce(output,label)
        
        out = output.max(1)[1].to('cpu').detach().numpy()
        label = label.to('cpu').detach().numpy()            
        train_loss_total += loss
        acc_train = accuracy_score(label,out)
        train_acc_total += acc_train
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = train_loss_total/i
        acc_train = train_acc_total/i
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(acc_train.item()))
        logger.info(f'Epoch: {epoch + 1}, loss_train: {loss.item()}, acc_train: {acc_train.item()}')
        acc_val,f1_val,precision_val,recall_val,loss_val = test(data,val_idx,model,loss_ce)
        print("Val set results:",
          "epoch= {:}".format(epoch+1),
          "test_accuracy= {:.4f}".format(acc_val),
          "precision= {:.4f}".format(precision_val),
          "recall= {:.4f}".format(recall_val),
          "f1_score= {:.4f}".format(f1_val))
        logger.info(f"Val set results:epoch={epoch+1}, val_accuracy= {acc_val}, precision= {precision_val}, recall= {recall_val}, f1_score= {f1_val}")
        # Save the model with the least loss on the validation set
        if acc_val > max_acc:
            max_acc = acc_val
            logger.info("save model...")
            print("save model...")
            # Save the model statement and store the status file, in which only the parameters are stored
            torch.save(model.state_dict(),"BotSim-24-Exp/Method/BestModel/{}.pth".format(model_file))
        # Test
        acc_test,f1_test,precision_test,recall_test,loss = test(data,test_idx,model,loss_ce)
        print("Test set results:",
          "epoch= {:}".format(epoch),
          "test_accuracy= {:.4f}".format(acc_test),
          "precision= {:.4f}".format(precision_test),
          "recall= {:.4f}".format(recall_test),
          "f1_score= {:.4f}".format(f1_test))
        
        logger.info(f"Test set results: epoch={epoch+1}, test_accuracy= {acc_test}, precision= {precision_test}, recall= {recall_test}, f1_score= {f1_test}")

        
    
        
# Test
@torch.no_grad()
def test(data,idx,model,loss_ce):
    model.eval()
    acc_test_total = 0
    f1_test_total = 0
    precision_test_total = 0
    recall_test_total = 0
    loss_test_total = 0
    i = 0
    
    data = data.cuda()
    value_feature = data.x[:,0:10].cuda()
    text_feature = data.x[:, 10:].cuda()        

    label = data.y.cuda()
    label = label[idx].long()
    output = model(value_feature, text_feature,data.edge_index,data.edge_weight, idx).float()

    loss = loss_ce(output,label)
    label = label.to('cpu').detach().numpy()
    out = output.max(1)[1].to('cpu').detach().numpy()       
    loss_test_total += loss
    acc_test = accuracy_score(label,out)
    acc_test_total += acc_test
    f1_test = f1_score(label,out)
    f1_test_total += f1_test
    precision_test = precision_score(label,out)
    precision_test_total += precision_test
    recall_test = recall_score(label,out)
    recall_test_total += recall_test
    i += 1
    acc_test = acc_test_total/i
    f1_test = f1_test_total/i
    precision_test = precision_test_total/i
    recall_test = recall_test_total/i
    loss = loss_test_total/i
    
    return acc_test,f1_test,precision_test,recall_test,loss
    

# main
if __name__ == "__main__":
    # Run time record: start time
    begin_time = time.localtime()
    begin_time = time.strftime('%Y-%m-%d %H:%M:%S',begin_time)
    print('Show the time when the program starts:',begin_time)
    
    # Parameter configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=[0,1,2,3,4], nargs='+', help='selection of random seeds')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--value_num', type=int, default=10, help='Number of value type feature.')
    parser.add_argument('--feature_dim', type=int, default=128, help='Number of attr dim.')
    parser.add_argument('--text_num', type=int, default=768, help='Number of tweet type feature.')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of batch size')
    parser.add_argument('--name', type=str, default='attr', help='name of logger')
    parser.add_argument('--log_dir', type=str, default='BotSim-24-Exp/Method/BestModel/gcn_log.log', help='dir of logger')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of gnn layers.')
    parser.add_argument('--model_file', type=str, default='rgcn', help='Save model dir.')

    args = parser.parse_args()
    logger = get_logger(args.name,args.log_dir)
    logger.info('test logger')
    logger.info(f'Show the time when the program starts:{begin_time}')

    # A total of 5 experiments were carried out and the average performance was obtained
    acc_list =[]
    precision_list = []
    recall_list = []
    f1_list = []
    for i,seed in enumerate(args.random_seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # Load data set
        print('load data...')
        dataloader,train_idx, val_idx, test_idx = load_data(seed)
        model = GCN(args.value_num, args.text_num,args.feature_dim,args.classes,args.dropout,dataloader,args.num_layers)

        model = model.cuda()
        print('begin the {} training'.format(i+1))
        logger.info(f'begin the {i+1} training')

        train(dataloader,train_idx, val_idx, test_idx,model,args.num_epochs,args.lr,args.weight_decay,logger, args.model_file)

        print('End the {} training'.format(i+1))
        logger.info(f'End the {i+1} training')
        # Load the model that works best, test it again, you need to create the model, and then you need to load these values into the model
        model.load_state_dict(torch.load('BotSim-24-Exp/Method/BestModel/{}.pth'.format(args.model_file)))
        acc_test,f1_test,precision_test,recall_test,loss = test(dataloader,test_idx,model,loss_ce)
        print("Test set Best results:",
          "test_accuracy= {:.4f}".format(acc_test),
          "precision= {:.4f}".format(precision_test),
          "recall= {:.4f}".format(recall_test),
          "f1_score= {:.4f}".format(f1_test))
        logger.info(f"Test set results: test_accuracy= {acc_test}, precision= {precision_test}, recall= {recall_test}, f1_score= {f1_test}")
        acc_list.append(acc_test*100)
        precision_list.append(precision_test*100)
        recall_list.append(recall_test*100)
        f1_list.append(f1_test*100)
    print('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    print('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    print('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    print('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list))) 
    
    
    logger.info('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    logger.info('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    logger.info('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    logger.info('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list))) 

    # Record the program end time
    end_time = time.localtime()
    end_time = time.strftime('%Y-%m-%d %H:%M:%S',end_time)
    print('Show when the program ends:',end_time)
    logger.info(f'Show when the program ends:{end_time}')
    startTime= datetime.datetime.strptime(begin_time,"%Y-%m-%d %H:%M:%S")
    endTime= datetime.datetime.strptime(end_time,"%Y-%m-%d %H:%M:%S")
    m,s = divmod((endTime- startTime).seconds,60)
    h, m = divmod(m, 60)
    print(f'Show the time consumed by the program:{h}h{m}m{s}s')
    logger.info(f'Show the time consumed by the program:{h}h{m}m{s}s')

