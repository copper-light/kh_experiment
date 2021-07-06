import torch
import matplotlib.pyplot as plt
import math
import numpy as np

class AverageMeter(object):
    """산술 평균 계산 펑션"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """모델의 top k 를 기준으로 정확도를 판단함"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


line_data_key = ["t_loss", "v_loss", "t_acc1", "v_acc1", "t_acc5", "v_acc5", "t_time", "v_time", "comunnication", 't_std_loss']
bar_data_key = ["train_val_time", "only_train_time"]

list_ylim = [[0, 6], [0,6], None, None, None,None,None,None,None, None]

def showChart(logs, output_file=None, ylim =None): # , ylim=None
    FONT_SIZE = 25
    plt.figure(figsize=(30, 70),  dpi=150)
    
    plt.rc('font', size=FONT_SIZE) # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE) # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE) # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE) # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE) # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE) # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE) # fontsize of the figure title
        
    if logs is None:
        pass
    
    position = 21 + (math.ceil(len(line_data_key) / 2 + 0.5) * 100)
        
    labels = [ log['label'] for log in logs ]

    epochs = len(logs[0]['values'][line_data_key[0]])
        
    for idx, key in enumerate(line_data_key):
        for log in logs:
            values = log['values']
            label = log['label'] 
            line = values[key]
            ylim = list_ylim[idx]

            plt.subplot(math.ceil(len(line_data_key) / 2 + 0.5), 2, idx+1)
            plt.plot(list(range(1, epochs + 1)), line, label=label)
            
            if (ylim is not None):
                if max(line) > ylim[1]:
                    plt.ylim(ylim[0], ylim[1]) 
                
            plt.ylabel(key)
            plt.xlabel('epoch')
            plt.legend()
       
    if output_file:
        plt.savefig(output_file)
        
    plt.show()

def createTimeChart(logs, output_file=None):
    x = np.array(range(len(logs)))
    labels = [ item['label']for item in logs ]
    values = [ int(item['values']['train_val_time']) for item in logs ]
    values2 = [ int(item['values']['only_train_time']) for item in logs ]

    FONT_SIZE = 10

    plt.rc('font', size=FONT_SIZE) # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE) # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE) # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE) # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE) # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE) # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE) # fontsize of the figure title

    plt.figure(figsize=(10, 5),  dpi=150)
    for idx, value in enumerate(values):
        plt.text(idx-0.15, value, value,
                 color='black',
                 horizontalalignment='center',
                 verticalalignment='bottom')
        plt.text(idx+0.15, values2[idx], values2[idx],
                 color='black',
                 horizontalalignment='center',
                 verticalalignment='bottom')

    plt.ylim(min(values2)* 0.9, max(values) * 1.03)
    plt.bar(x-0.15, values, width=0.3, label="total_time")
    plt.bar(x+0.15, values2, width=0.3, label="only_train_time")
    plt.xticks(x, labels)
    plt.legend()
    
    if output_file:
        plt.savefig(output_file)
        
    plt.show()
    
    
def parseLog(path):
    t_loss = []
    t_acc1 = []
    t_acc5 = []
    t_time = []
    t_std_loss = []
    
    s_loss = []
    s_acc1 = []
    s_acc5 = []
    s_time = []
    s_std_loss = []
    
    v_loss = []
    v_acc1 = []
    v_acc5 = []
    v_time = []

    comunnication = []
    train_val_time = 0
    only_train_time =0
    with open(path, 'r', encoding='utf-8') as f:
        for row in f.readlines():
            item = row.split(',')

            if row.find('val time') > -1: # len 6
                #2021-05-02 22:49:02,017:INFO:[Rank-0] val time 12.1949, epoch  99, step  27, loss 2.56235504, auc@1 50.53, auc@5 79.07
                item[0] # time
                v_time.append(float(item[1].strip().split(' ')[3])) # duration
                item[2].split(' ')[1] # epoch
                item[3].split(' ')[1] # step
                v_loss.append(float(item[4].strip().split(' ')[1])) # loss
                v_acc1.append(float(item[5].strip().split(' ')[1])) # acc@1
                v_acc5.append(float(item[6].strip().split(' ')[1])) # acc@5

            elif row.find('train time') > -1: # len 7
                #2021-05-20 00:42:09,945:INFO:[Rank-0] train time 39.6154, epoch  26, step  98, loss 3.43209100, auc@1 17.13, auc@5 43.60, share 48, std_loss 0.02590131636100764,  3.40098 3.44795 3.43573 3.44370
                item[0] # time
                t_time.append(float(item[1].strip().split(' ')[3])) # duration
                item[2].split(' ')[1] # epoch
                item[3].split(' ')[1] # step
                t_loss.append(float(item[4].strip().split(' ')[1])) # loss
                t_acc1.append(float(item[5].strip().split(' ')[1])) # acc@1
                t_acc5.append(float(item[6].strip().split(' ')[1])) # acc@5
                comunnication.append(int(item[7].strip().split(' ')[1])) # share
                
                t_std_loss.append(float(item[8].strip().split(' ')[1])) # std_loss
                
            elif row.find('step time') > -1:
                item[0] # time
                s_time.append(float(item[1].strip().split(' ')[3])) # duration
                item[2].split(' ')[1] # epoch
                item[3].split(' ')[1] # step
                s_loss.append(float(item[4].strip().split(' ')[1])) # loss
                s_acc1.append(float(item[5].strip().split(' ')[1])) # acc@1
                s_acc5.append(float(item[6].strip().split(' ')[1])) # acc@5
                
                s_std_loss.append(float(item[8].strip().split(' ')[1])) # share
                
            
            elif row.find('finish') > -1:
                item = row.strip().split(',')
                train_val_time = item[2].strip().split(' ')[1]
                only_train_time = item[3].strip().split(' ')[1]
                train_val_time = float(train_val_time)
                only_train_time = float(only_train_time)
                
    return {
        't_loss':t_loss, 't_acc1':t_acc1, 't_acc5':t_acc5, 't_time': t_time, 
        'v_loss': v_loss, 'v_acc1':v_acc1, 'v_acc5':v_acc5, 'v_time': v_time, 
        'comunnication':comunnication, 'train_val_time': train_val_time, 'only_train_time': only_train_time, 't_std_loss': t_std_loss, 
        's_loss':s_loss, 's_acc1':s_acc1, 's_acc5':s_acc5, 's_time': s_time}


def create_chart(path):
    proposal = f"{path}/proposal_0.log"
    minibatch = f"{path}/minibatch_0.log"
    localsgd = f"{path}/local_0.log"
    pro = parseLog(proposal)
    mb = parseLog(minibatch)
    lo = parseLog(localsgd)

    # train time
    print(mb['train_val_time'], lo['train_val_time'], pro['train_val_time'])
    print(mb['only_train_time'], lo['only_train_time'], pro['only_train_time'])

    epoch = len(mb['t_loss'])
    labels = ["minibatch", "local sgd", "proposal"]

    # loss
    showChart(["t_loss","v_loss"], epoch, 
            values = [
                [mb['t_loss'], lo['t_loss'], pro['t_loss']], 
                [mb['v_loss'], lo['v_loss'], pro['v_loss']]
            ], 
            labels = labels,
            ylim =[0,8], output_file=f"{path}/chart_loss.png")
    
    # acc1
    showChart(["t_acc1","v_acc1"], epoch, 
            values = [
                [mb['t_acc1'], lo['t_acc1'], pro['t_acc1']], 
                [mb['v_acc1'], lo['v_acc1'], pro['v_acc1']]
            ],  
            labels = labels, output_file=f"{path}/chart_acc1.png")

    # acc5
    showChart(["t_acc5","v_acc5"], epoch, 
            values = [
                [mb['t_acc5'], lo['t_acc5'], pro['t_acc5']], 
                [mb['v_acc5'], lo['v_acc5'], pro['v_acc5']]
            ],  
            labels = labels, output_file=f"{path}/chart_acc5.png")

    # time
    showChart(["t_time","v_time"], epoch, 
            values = [
                [mb['t_time'], lo['t_time'], pro['t_time']], 
                [mb['v_time'], lo['v_time'], pro['v_time']]
            ],   
            labels = labels, output_file=f"{path}/chart_time.png")

    # allreduce
    showChart(["comunnication"], epoch, 
            values = [
                [mb['comunnication'], lo['comunnication'], pro['comunnication']]
            ], labels = labels, output_file=f"{path}/chart_allreduce.png")

