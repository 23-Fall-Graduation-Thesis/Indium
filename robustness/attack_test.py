import  torch, argparse, csv
from tqdm import tqdm
import attack_utils

from dataset_dir.datasets import datasetload

def main(args):
    attack_utils.set_seed()
    device = torch.device('cuda:'+str(args.device_num))

    d_len = {'cifar10':10000, 'cifar100':10000, 'svhn':10000, 'cub':5994}

    # load dataset (use test set)
    _, _, test_loader, n_way = datasetload(args.dataset)

    # load model
    model = attack_utils.setModel(args.model, n_way, args.imgsz, args.imgc, pretrained=None)
    model.load_state_dict(torch.load(f'./model/weight/custom/alexnet/{args.dataset}_epoch100_lr0.001_freeze{args.freeze_option}111.pt'))
    model = model.to(device)

    at = attack_utils.setAttack(args.test_attack, model, args.test_eps, args)

    model.eval()
    clean, adver = 0, 0
    for _, (x, y) in enumerate(tqdm(test_loader)):
        x = x.to(device)
        y = y.to(device)

        # calculate performance of clean images
        correct_count, _ = attack_utils.predict(x, y, model)
        clean += correct_count
        
        # adversarial attack
        advx = at.perturb(x, y)
        correct_count, _ = attack_utils.predict(advx, y, model)
        adver += correct_count

    acc = round(100.0*clean/d_len[args.dataset], 4)
    adv_acc = round(100.0*adver/d_len[args.dataset], 4)
    
    result = [args.freeze_option, args.dataset, args.test_attack, args.test_eps, acc, adv_acc]
    with open('attack_performance.csv', 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(result)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Dataset options
    argparser.add_argument('--dataset', type=str, help='dataset', default="cifar10")
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=224)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    
    # Training options
    argparser.add_argument('--model', type=str, help='type of model to use', default="alexnet")
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.01)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--test_eps', type=float, help='attack-eps', default=8.0)
    argparser.add_argument('--iter', type=int, default=10)

    argparser.add_argument('--freeze_option', type=str, default="11001")

    args = argparser.parse_args()

    main(args)