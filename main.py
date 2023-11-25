import torch, os
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import argparse
from model.models import Conv4
from model.pretrained_models import get_pretrain_model
from dataset_dir.datasets import datasetload
from finetune import *
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_parse(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', help='Dataset type')
    parser.add_argument('--model', default='Conv4', help='Model type')
    parser.add_argument('--pretrain', type=str2bool, nargs='?', const=True, default=False, help="Pretrain")
    parser.add_argument('--mode', type=str, default='cus', help='Custom(cus) or Optimal(opt)')
    parser.add_argument('--freeze', type=str, default='11111111', help='Layer freezing 0 : freeze, 1 : not freeze')
    
    parser.add_argument('--epoch', type=int, default=50, help='Epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')

    return parser.parse_args()

def select_model(name):
    if name == 'Conv4':
        return Conv4()
    else:
        raise ValueError(f'Invalid mode name.')

def train(model, criterion, optimizer, trainloader, device):
    model.train()
    
    train_loss = 0
    loss = 0
    train_acc = 0
    for data, target in tqdm(trainloader, desc="train"):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_acc += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
    
    train_loss = train_loss / len(trainloader.dataset)
    train_acc = train_acc / len(trainloader.dataset) * 100
    
    return train_loss, train_acc

def validation(model, criterion, validloader, device):
    model.eval()
    
    valid_loss = 0
    valid_acc = 0
    with torch.no_grad():
        for data, target in tqdm(validloader, desc="val"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            valid_acc += pred.eq(target.view_as(pred)).sum().item()
    
    valid_loss = valid_loss / len(validloader.dataset)
    valid_acc = valid_acc / len(validloader.dataset) * 100
    
    return valid_loss, valid_acc
    
def test(model, criterion, testloader, checkpt, device):
    model.load_state_dict(torch.load(checkpt))
    model.eval()
    
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            test_acc += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss = test_loss / len(testloader.dataset)
    test_acc = test_acc / len(testloader.dataset) * 100
    
    return test_loss, test_acc

if __name__ == '__main__':
    # arguments parsing
    args = arg_parse(argparse.ArgumentParser())
    
    # random seed
    np.random.seed(2023)
    torch.manual_seed(2023)
    
    # cuda device
    conf = dict()
    conf['device'] = torch.device("cuda:" + str(args.device))
    conf = dict(conf, **args.__dict__)
    
    # dataset load
    trainloader, validloader, testloader, num_class = datasetload(conf['dataset'])
    
    # experiment setting values
    setting = "epoch"+str(conf['epoch'])+"_lr"+str(conf['lr'])
    
    if conf['pretrain'] == True:
        model = select_model(conf['model']).to(conf['device'])
        checkpt = "./model/weight/pretrain/"+str(conf['model'])+"/"+str(conf['dataset'])+"_"+setting+".pt"
        board_name = str(conf['model'])+"/"+str(conf['dataset'])+"_"+setting
        writer = SummaryWriter("./results/log/pretrain/"+board_name)
        
        print('model:', conf['model'], ' dataset:', conf['dataset'], 'pretrain: ', conf['pretrain'])
    else:
        # pretrained model load
        model = get_pretrain_model(conf['model'], num_class).to(conf['device'])
        
        # save name setting & model layer selection
        if conf['mode'] == 'cus': # custom mode
            layers = conf['freeze']
            model = custom_finetuning(model, layers).to(conf['device'])
            checkpt = "./model/weight/custom/"+str(conf['model'])+"/"+str(conf['dataset'])+"_"+setting+".pt"
            board_name = str(conf['model'])+"/"+str(conf['dataset'])+"_"+setting+"_freeze"+layers
            writer = SummaryWriter("./results/log/custom/"+board_name)
            os.makedirs("./model/weight/custom/"+str(conf['model'])+"/", exist_ok=True)
        elif conf['mode'] == 'opt': # optimal mode
            model = optimal_finetuning(model).to(conf['device'])
            checkpt = "./model/weight/optimal/"+str(conf['model'])+"/"+str(conf['dataset'])+"_"+setting+".pt"
            board_name = str(conf['model'])+"/"+str(conf['dataset'])+"_"+setting
            writer = SummaryWriter("./results/log/optimal/"+board_name)
            os.makedirs("./model/weight/optimal/"+str(conf['model'])+"/", exist_ok=True)
        else:
            raise ValueError(f'Invalid finetune mode input.')
        
        print('model:', conf['model'], ' dataset:', conf['dataset'], 'fine-tuning mode:', conf['mode'], 'freeze: ', conf['freeze'])
    
    print('Experiment Setting:', setting, '|Croess-Entropy Loss|SGD optimizer')
    
    # loss, optimizer define
    criterion = nn.CrossEntropyLoss().to(conf['device'])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['lr'])
    
    print('\nStart training')
    best = 99999999
    best_epoch = 0
    bad_count = 0
    for epoch in range(conf['epoch']):
        train_loss, train_acc = train(model, criterion, optimizer, trainloader, conf['device'])
        valid_loss, valid_acc = validation(model, criterion, validloader, conf['device'])
        if (epoch + 1) % 5 == 0:
            print('Epoch:{:04d}'.format(epoch+1), 'train loss:{:.3f}'.format(train_loss), 'acc:{:.2f}'.format(train_acc))
            print('validation loss:{:.3f}'.format(valid_loss), 'acc:{:.2f}'.format(valid_acc))
        if valid_loss < best:
            best = valid_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpt)
            bad_count = 0
        else:
            bad_count += 1
        
        if bad_count == 30:
            break
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        
        writer.add_scalar('Loss/val', valid_loss, epoch)
        writer.add_scalar('Acc/val', valid_acc, epoch)
        
    print('Finish')
    print('start test')
    
    test_loss, test_acc = test(model, criterion, testloader, checkpt, conf['device'])
    print('Load {}th epoch'.format(best_epoch))
    print('test loss:{:.3f}'.format(test_loss), 'acc:{:.2f}'.format(test_acc))