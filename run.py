import argparse
import collections
import itertools
import torch
import random
import torch.nn.functional as F
from torch.autograd import Variable
from net import Vaiaf
from get_indicator_matrix_A import get_mask
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config

#source

def main(MR=[0.3]):
    # Environments

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    print("GPU: " + str(use_cuda))
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    for key in dataset:
        print("\n\n")
        print("======================================================================")
        config = get_default_config(dataset[key])
        config['dataset'] = dataset[key]
        if config['dataset'] == 'NoisyMNIST':
            MR = [0.3]
        # print(key, ":", dict_1[key])
        print("Data set: " + (dataset[key]))
        config['print_num'] = config['training']['epoch'] / 10  # print_num
        logger = get_logger()

        # Load data
        seed = config['training']['seed']
        X_list, Y_list = load_data(config)
        x1_train_raw = X_list[0]
        x2_train_raw = X_list[1]



        for missingrate in MR:
            accumulated_metrics = collections.defaultdict(list)
            config['training']['missing_rate'] = missingrate
            print('--------------------Missing rate = ' + str(missingrate) + '--------------------')
            for data_seed in range(1, args.test_time + 1):
                # get the mask
                np.random.seed(data_seed)
                mask = get_mask(2, x1_train_raw.shape[0], config['training']['missing_rate'])
                # mask the data
                x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
                x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]

                x1_train = torch.from_numpy(x1_train).float().to(device)
                x2_train = torch.from_numpy(x2_train).float().to(device)
                mask = torch.from_numpy(mask).long().to(device)  # indicator matrix A

                # Set random seeds for model initialization
                np.random.seed(seed)
                random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = True

                # Build the model
                APADC = Apadc(config)
                optimizer = torch.optim.Adam(
                    itertools.chain(APADC.autoencoder1.parameters(), APADC.autoencoder2.parameters()),
                    lr=config['training']['lr'])
                APADC.to_device(device)

                # Print the models
                # logger.info(APADC.autoencoder1)
                # logger.info(APADC.autoencoder2)
                # logger.info(optimizer)

                # # init p distribution
                # p_sample = np.ones(2)
                # weight_history = []
                # p_sample = p_sample / sum(p_sample)
                # p_sample = torch.FloatTensor(p_sample).cuda()
                #
                # # init adaptive weight
                # adaptive_weight = np.ones(2)
                # adaptive_weight = adaptive_weight / sum(adaptive_weight)
                # adaptive_weight = torch.FloatTensor(adaptive_weight).cuda()
                # adaptive_weight = adaptive_weight.unsqueeze(1)


                # Training
                flag_1 = (torch.LongTensor([1, 1]).to(device) == mask).int()
                # flag = (torch.LongTensor([1, 1]).to(device) == mask).int()
                # flag_1 = (flag_1[:, 1] + flag_1[:, 0]) == 2
                Y_list1 = torch.tensor(Y_list).int().to(device).squeeze(dim=0).unsqueeze(dim=1)
                Tmp_acc, Tmp_nmi, Tmp_ari = 0, 0, 0
                for epoch in range(config['training']['epoch'] + 1):
                    if epoch <= config['training']['epoch']:
                    # if epoch < config['training']['epoch']:
                        Tmp_acc, Tmp_nmi, Tmp_ari =APADC.train(config, logger, x1_train, x2_train, Y_list, mask, optimizer, device)
                        # APADC.train(config, logger, x1_train, x2_train, mask, optimizer, device,flag_1,epoch,Y_list1, p_sample, adaptive_weight)


                accumulated_metrics['acc'].append(Tmp_acc)
                accumulated_metrics['nmi'].append(Tmp_nmi)
                accumulated_metrics['ari'].append(Tmp_ari)
                # accumulated_metrics['pur'].append(Tmp_pur)
                print('------------------------Training over------------------------')

            cal_std(logger, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ari'])
            # cal_std(logger, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ari'], accumulated_metrics['pur'])


if __name__ == '__main__':
    dataset = {
        # 0: "MNIST-USPS",
        #        1: "Caltech101-20"
    }
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=int, default=str(1), help='dataset id')  # data index
    parser.add_argument('--test_time', type=int, default=str(1), help='number of test times')
    parser.add_argument('--devices', type=str, default='1', help='gpu device ids')
    args = parser.parse_args()
    # dataset = dataset[args.dataset]
    MisingRate = [0.1, 0.3, 0.5, 0.7]
    main(MR=MisingRate)

