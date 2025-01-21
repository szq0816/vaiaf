from einops import rearrange
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.nn.functional import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from loss import Loss, Proto_Align_Loss, Instance_Align_Loss, DeepMVCLoss, MIA, DDA
import evaluation
from util import next_batch,next_batch1,get_Similarity
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda:0' if use_cuda else 'cpu')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,config,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, feature Z^v.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, feature Z^v.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction samples.
        """
        x_hat = self._decoder(latent)
        return x_hat



    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, feature Z^v.
              x_hat:  [num, feat_dim] float tensor, reconstruction samples.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

class Discriminator(nn.Module):
    def __init__(self,
                 encoder_dim,
                 ):
        super(Discriminator, self).__init__()
        self.l = nn.Sequential(
            nn.Linear(encoder_dim[0], 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def discriminator(self, x):
        return self.l(x)



class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent




# class MultiHeadAttention(nn.Module):
#     def __init__(self, attention_dropout_rate, num_heads, attn_bias_dim):
#         super(MultiHeadAttention, self).__init__()
#
#         self.attention_dropout_rate = attention_dropout_rate
#         self.num_heads = num_heads
#         self.attn_bias_dim = attn_bias_dim
#
#     def forward(self, q, k, v):
#         batch_size, hidden_size = q.size()
#
#         att_size = hidden_size // self.num_heads
#         scale = att_size ** -0.5
#         a = self.num_heads * att_size
#         # Check if linear layers need to be initialized
#         self.linear_q = nn.Linear(hidden_size, self.num_heads * att_size).to(device)
#         self.linear_k = nn.Linear(hidden_size, self.num_heads * att_size).to(device)
#         self.linear_v = nn.Linear(hidden_size, self.num_heads * att_size).to(device)
#         self.linear_bias = nn.Linear(self.attn_bias_dim, self.num_heads).to(device)
#         self.att_dropout = nn.Dropout(self.attention_dropout_rate).to(device)
#
#         self.output_layer = nn.Linear(self.num_heads * att_size, 1).to(device)
#
#
#         q = self.linear_q(q).view(batch_size, -1, self.num_heads, att_size)
#         k = self.linear_k(k).view(batch_size, -1, self.num_heads, att_size)
#         v = self.linear_v(v).view(batch_size, -1, self.num_heads, att_size)
#
#         q = q.transpose(1, 2)  # [batch_size, num_heads, q_len, att_size]
#         v = v.transpose(1, 2)  # [batch_size, num_heads, q_len, att_size]
#         k = k.transpose(1, 2).transpose(2, 3)  # [batch_size, num_heads, att_size, q_len]
#
#         q = q * scale
#         x = torch.matmul(q, k)  # [batch_size, num_heads, q_len, q_len]
#
#         x = torch.softmax(x, dim=3)
#         x = self.att_dropout(x)
#         x = x.matmul(v)  # [batch_size, num_heads, q_len, att_size]
#
#         x = x.transpose(1, 2).contiguous()  # [batch_size, q_len, num_heads, att_size]
#         x = x.view(batch_size, self.num_heads * att_size)  # [batch_size, q_len, num_heads * att_size]
#
#         x = self.output_layer(x)
#
#         return x
#
#
# class FeedForwardNetwork(nn.Module):
#     def __init__(self, view, ffn_size):
#         super(FeedForwardNetwork, self).__init__()
#
#         self.layer1 = nn.Linear(view, ffn_size)
#         self.gelu = nn.GELU()
#         self.layer2 = nn.Linear(ffn_size, view)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.gelu(x)
#         x = self.layer2(x)
#         x = x.unsqueeze(1)
#         return x


class Apadc():
    """Apadc module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']

        self.autoencoder1 = Autoencoder(config, config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config, config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])

        self.discriminator1 = Discriminator(config['Autoencoder']['arch1'])
        self.discriminator2 = Discriminator(config['Autoencoder']['arch2'])

        self.pre1 = Prediction(self._dims_view1)
        self.pre2 = Prediction(self._dims_view2)
        # self.label_contrastive_module = nn.Sequential(
        #     nn.Linear(config['Autoencoder']['arch1'][-1], config['label']),
        #     nn.Softmax(dim=1)
        # )
        self.label_learning_module = nn.Sequential(
            nn.Linear(config['Autoencoder']['arch1'][-1], config['training']['high_dim']),
            nn.Linear(config['training']['high_dim'], config['label']),
            nn.Softmax(dim=1)
        )
        self.view = 2
        self.cross_attn = cross_attn(dim=config['training']['high_dim'], num=config['label'], head=1).cuda()
        # self.attention_dropout_rate = 0.5
        # self.num_heads = 6
        # self.attn_bias_dim = 6
        # self.ffn_size = 32
        # self.threshold = 0.8
        # self.attention_net = MultiHeadAttention(self.attention_dropout_rate, self.num_heads,
        #                                    self.attn_bias_dim)
        # self.p_net = FeedForwardNetwork(2, self.ffn_size)
        # self.Specific_view = nn.Sequential(
        #     nn.Linear(config['Autoencoder']['arch1'][-1], config['training']['high_dim']),
        # )



    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)
        self.discriminator1.to(device)
        self.discriminator2.to(device)
        self.label_learning_module.to(device)
        self.pre1.to(device)
        self.pre2.to(device)
        # self.p_net.to(device)
        # self.attention_net.to(device)
        # self.agg.to(device)
        # self.Specific_view.to(device)



    def train(self, config, logger, x1_train, x2_train, mask, optimizer, device,flag_1,epoch,Y_list,Tmp_acc, Tmp_nmi, Tmp_ari, lambda1, lambda2, lambda3):
        """Training the model.
            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari
        """


        # criterion = DeepMVCLoss(config['samples'], config['label']).to(device)
        criterion = Loss(config['training']['batch_size'], config['label'], config['training']['temperature_f'],
                         config['training']['temperature_l']).to(device)
        X1, X2, X3, X4 = shuffle(x1_train, x2_train, flag_1[:, 0], flag_1[:, 1])
        loss_all, loss_rec1, loss_rec2, loss_mia, loss_dda, gan_loss = 0, 0, 0, 0, 0, 0
        for batch_x1, batch_x2, x1_index, x2_index, batch_No in next_batch(X1, X2, X3, X4,
                                                                           config['training']['batch_size']):
            if len(batch_x1) == 1:
                continue
            index_both = x1_index + x2_index == 2  # C in indicator matrix A of complete multi-view data
            index_peculiar1 = (
                    x1_index + x1_index + x2_index == 2)  # I^1 in indicator matrix A of incomplete multi-view data
            index_peculiar2 = (
                    x1_index + x2_index + x2_index == 2)  # I^2 in indicator matrix A of incomplete multi-view data
            z_1 = self.autoencoder1.encoder(batch_x1[x1_index == 1])  # [Z_C^1;Z_I^1]
            z_2 = self.autoencoder2.encoder(batch_x2[x2_index == 1])  # [Z_C^2;Z_I^2]

            hz1, hz2, S = self.cross_attn(z_1, z_2)
            # sg_loss = []
            # # z1, z2, S = self.cross_attn(z_1, z_2)
            # # loss_reg = attention_reg(S)
            # sg_loss.append(criterion.Structure_guided_Contrastive_Loss(z_1, hz1, S))
            # sg_loss.append(criterion.Structure_guided_Contrastive_Loss(z_2, hz2, S))
            # sg_loss = sum(sg_loss)
            #
            recon1 = F.mse_loss(self.autoencoder1.decoder(hz1), batch_x1[x1_index == 1])
            recon2 = F.mse_loss(self.autoencoder2.decoder(hz2), batch_x2[x2_index == 1])

            # recon1 = F.mse_loss(self.autoencoder1.decoder(z_1), batch_x1[x1_index == 1])
            # recon2 = F.mse_loss(self.autoencoder2.decoder(z_2), batch_x2[x2_index == 1])
            rec_loss = (recon1 + recon2)  # reconstruction losses \sum L_REC^v
            # Discriminator
            # d_loss = []
            fake_z1 = self.discriminator1.discriminator(
                self.autoencoder1.decoder(z_1))
            real_z1 = self.discriminator1.discriminator(batch_x1[x1_index == 1])
            fake_z2 = self.discriminator2.discriminator(
                self.autoencoder2.decoder(z_2))
            real_z2 = self.discriminator2.discriminator(batch_x2[x2_index == 1])
            # rec_loss = -(torch.mean(fake_z1) * 0.5 + torch.mean(real_z1) * 0.5)
            d_loss = (torch.mean(nn.ReLU(inplace=True)(1.0 - real_z1)) + torch.mean(
                nn.ReLU(inplace=True)(1.0 + fake_z1))) + \
                     (torch.mean(nn.ReLU(inplace=True)(1.0 - real_z2)) + torch.mean(
                         nn.ReLU(inplace=True)(1.0 + fake_z2)))
            gan_loss = rec_loss*lambda2 + d_loss*lambda3

            # ##align
            # z1 = self.autoencoder1.encoder(batch_x1)
            # z2 = self.autoencoder2.encoder(batch_x2)
            # z = [z1, z2]
            # miss_vec = [x1_index, x2_index]
            # q1 = self.label_learning_module(z1)
            # q2 = self.label_learning_module(z2)
            # loss_list = list()
            # loss_list.append(
            #     0.01 * criterion.forward_label(q1, q2, config['training']['temperature_f'], normalized=False))
            # loss_list.append(0.01 * criterion.forward_prob(q1, q2))
            # algin_loss = sum(loss_list)
            # ###### 计算loss_ins_align ######
            # criterion_ins = Instance_Align_Loss().to(device)
            # loss_list_ins = []
            # for v1 in range(self.view):
            #     v2_start = v1 + 1
            #     for v2 in range(v2_start, self.view):
            #         align_index = []
            #         for i in range(batch_x1.shape[0]):
            #             if miss_vec[v1][i] == 1 and miss_vec[v2][i] == 1:
            #                 align_index.append(i)
            #         z1 = z[v1][align_index]  # 改
            #         z2 = z[v2][align_index]  # 改
            #         Dx = F.cosine_similarity(z1, z2, dim=1)
            #         gt = torch.ones(z1.shape[0]).to(device)
            #         l_tmp2 = criterion_ins(gt, Dx)
            #         loss_list_ins.append(l_tmp2)
            # loss_ins_align = sum(loss_list_ins) * 1e-3
            #
            # criterion_proto = Proto_Align_Loss().to(device)
            # loss_list_pro = []
            # for v1 in range(self.view):
            #     v2_start = v1 + 1
            #     for v2 in range(v2_start, self.view):
            #         align_index = []
            #         for i in range(z[0].shape[0]):
            #             if miss_vec[v1][i] == 1 and miss_vec[v2][i] == 1:
            #                 align_index.append(i)
            #         p1 = z[v1][align_index].t()
            #         p2 = z[v2][align_index].t()
            #         gt = torch.ones(p1.shape[0]).to(device)
            #         Dp = F.cosine_similarity(p1, p2, dim=1)
            #         # Dp = get_Similarity(p1, p2)
            #         l_tmp = criterion_proto(gt, Dp)
            #         # l_tmp = F.mse_loss(gt, Dp)
            #         loss_list_pro.append(l_tmp)
            # loss_pro_align = sum(loss_list_pro) * 1e-3
            #
            # align_loss = algin_loss + loss_pro_align

            # hb1, hb2, _ = self.cross_attn(z_view1_both, z_view2_both)
            # # Cross-view Dual-Prediction Loss
            # img2txt, _ = self.pre1(z_view1_both)
            # txt2img, _ = self.pre2(z_view2_both)
            # recon3 = F.mse_loss(img2txt, z_view2_both)
            # recon4 = F.mse_loss(txt2img, z_view1_both)
            # dualprediction_loss = (recon3 + recon4)


            z_view1_both = self.autoencoder1.encoder(batch_x1[index_both])
            z_view2_both = self.autoencoder2.encoder(batch_x2[index_both])

            if len(batch_x2[index_peculiar2]) % config['training']['batch_size'] == 1:
                continue
            z_view2_peculiar = self.autoencoder2.encoder(batch_x2[index_peculiar2])
            if len(batch_x1[index_peculiar1]) % config['training']['batch_size'] == 1:
                continue
            z_view1_peculiar = self.autoencoder1.encoder(batch_x1[index_peculiar1])

            w1 = torch.var(z_view1_both)
            w2 = torch.var(z_view2_both)
            a1 = w1 / (w1 + w2)
            a2 = 1 - a1
            # the weight matrix is only used in MMI loss to explore the common cluster information
            # z_i = \sum a_iv w_iv z_i^v, here, w_iv = var(Z^v)/(\sum a_iv var(Z^v)) for MMI loss
            Z = torch.add(z_view1_both * a1, z_view2_both * a2)
            # mutual information losses \sum L_MMI^v (Z_C, Z_I^v)
            mia_loss = MIA(z_view1_both, Z) + MIA(z_view2_both, Z)

            view1 = torch.cat([z_view1_both, z_view1_peculiar, z_view2_peculiar], dim=0)
            view2 = torch.cat([z_view2_both, z_view1_peculiar, z_view2_peculiar], dim=0)
            # z_i = \sum a_iv w_iv z_i^v, here, w_iv = 1/\sum a_iv for MMD loss
            view_both = torch.add(view1, view2).div(2)
            # mean discrepancy losses   \sum L_MMD^v (Z_C, Z_I^v)
            dda_loss = DDA(view1, view_both, kernel_mul=config['training']['kernel_mul'],
                           kernel_num=config['training']['kernel_num']) + \
                       DDA(view2, view_both, kernel_mul=config['training']['kernel_mul'],
                           kernel_num=config['training']['kernel_num'])
            # total loss
            loss = mia_loss + dda_loss * lambda1 + gan_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            loss_rec1 += recon1.item()
            loss_rec2 += recon2.item()
            loss_dda += dda_loss.item()
            loss_mia += mia_loss.item()
            # gl_loss += gl_loss.item()
            # gan_loss += gan_loss.item()

        if (epoch) % config['print_num'] == 0:
            output = "Epoch: {:.0f}/{:.0f} " \
                     "==> REC loss = {:.4f} " \
                     "==> MMD loss = {:.4f} " \
                     "==> MMI loss = {:.4e} " \
                .format(epoch, config['training']['epoch'], (loss_rec1 + loss_rec2), loss_mmd, loss_mmi)
            # output = "Epoch: {:.0f}/{:.0f} " \
            #     .format(epoch, config['training']['epoch'])
            print(output)
            # if epoch >= config['training']['epoch'] * (1 / 2):
                # if epoch > config['training']['start_dual_prediction']:
            scores = self.evaluation(config, logger, mask, x1_train, x2_train, Y_list, device, epoch)
            if scores['kmeans']['ACC'] >= Tmp_acc:
                Tmp_acc = scores['kmeans']['ACC']
                Tmp_nmi = scores['kmeans']['NMI']
                Tmp_ari = scores['kmeans']['ARI']

        return Tmp_acc, Tmp_nmi, Tmp_ari

    def evaluation(self, config, logger, mask, x1_train, x2_train, Y_list, device, epoch):
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval()
            # self.pre1.eval(), self.pre2.eval()

            flag = mask[:, 0] + mask[:, 1] == 2  # complete multi-view data
            view2_missing_idx_eval = mask[:, 0] == 0  # incomplete multi-view data
            view1_missing_idx_eval = mask[:, 1] == 0  # incomplete multi-view data

            common_view1 = x1_train[flag]
            common_view1 = self.autoencoder1.encoder(common_view1)
            common_view2 = x2_train[flag]
            common_view2 = self.autoencoder2.encoder(common_view2)
            y_common = Y_list[flag]

            view1_exist = x1_train[view1_missing_idx_eval]
            view1_exist = self.autoencoder1.encoder(view1_exist)
            y_view1_exist = Y_list[view1_missing_idx_eval]
            view2_exist = x2_train[view2_missing_idx_eval]
            view2_exist = self.autoencoder2.encoder(view2_exist)
            y_view2_exist = Y_list[view2_missing_idx_eval]

            # view2_pre_exist, _ = self.pre1(view1_exist)
            # view1_pre_exist, _ = self.pre2(view2_exist)
            # a11 = config['a1']
            # a21 = 1 - a11
            # # view1_exist = torch.add(view1_exist, view2_pre_exist).div(2)
            # view1_exist = torch.add(view1_exist * a11, view2_pre_exist * a21)
            # # view1_exist = torch.add(view1_exist * 0.9, view2_exist * (1-0.9))
            # a12 = config['a1']
            # a22 = 1 - a12
            # # view2_exist = torch.add(view2_exist, view1_pre_exist).div(2)
            # view2_exist = torch.add(view2_exist * a12, view1_pre_exist * a22)


            # w1 = torch.var(view1_exist)
            # w2 = torch.var(view2_pre_exist)
            # a1 = w1 / (w1 + w2)
            # a2 = 1 - a1
            # view1_exist = torch.add(view1_exist * a1, view2_pre_exist * a2)
            # # a12 = config['a1']
            # # a22 = 1 - a12
            # w1 = torch.var(view2_exist)
            # w2 = torch.var(view1_pre_exist)
            # a1 = w1 / (w1 + w2)
            # a2 = 1 - a1
            # view2_exist = torch.add(view2_exist * a1, view1_pre_exist * a2)




            # (2) z_i = \sum a_iv w_iv z_i^v, here, w_iv = 1/\sum a_iv
            common = torch.add(common_view1, common_view2).div(2)

            latent_fusion = torch.cat([common, view1_exist, view2_exist], dim=0).cpu().detach().numpy()
            Y_list = torch.cat([y_common, y_view1_exist, y_view2_exist], dim=0).cpu().detach().numpy()

            scores, _ = evaluation.clustering([latent_fusion], Y_list[:, 0])
            # print("\033[2;29m" + 'Common features ' + str(scores) + "\033[0m")
            self.autoencoder1.train(), self.autoencoder2.train()
            # self.pre1.train(), self.pre2.train()
        return scores

class cross_attn(nn.Module):
    def __init__(self, dim=128, num=10, head=1):
        super(cross_attn, self).__init__()
        self.scale = dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj1 = nn.Linear(dim, dim, bias=True)
        self.v_proj2 = nn.Linear(dim, dim, bias=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.num_heads = head
        self.dim = dim
        self.num = num
        self.proj_z = nn.Identity()
        self.proj_c = nn.Identity()

    def forward(self, z1, z2):
        # B = z1.size(0)
        q = rearrange(
            self.q_proj(self.norm1(z1)),
            "n (h c)-> h n c",
            h=self.num_heads,
            n=z1.size(0),
            c=self.dim // self.num_heads,
        )
        k = rearrange(
            self.k_proj(self.norm2(z2)),
            "n (h c)-> h n c",
            n=z2.size(0),
            h=self.num_heads,
            c=self.dim // self.num_heads,
        )
        v = rearrange(
            self.v_proj1(self.norm2(z2)),
            "n (h c)-> h n c",
            n=z2.size(0),
            h=self.num_heads,
            c=self.dim // self.num_heads,
        )
        v1 = rearrange(
            self.v_proj2(self.norm1(z1)),
            "n (h c)-> h n c",
            h=self.num_heads,
            n=z1.size(0),
            c=self.dim // self.num_heads,
        )
        sim = (q @ k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim=-1)
        z_c = rearrange(
            attn @ v,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=z1.size(0),
            c=self.dim // self.num_heads,
        )
        c_z = rearrange(
            attn.transpose(-2, -1) @ v1,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=z2.size(0),
            c=self.dim // self.num_heads,
        )
        z_c = self.proj_z(z_c)
        c_z = self.proj_c(c_z)

        # w11 = torch.var(z_c)
        # w12 = torch.var(z1)
        # a11 = w11 / (w11 + w12)
        # a12 = 1 - a11
        # z_c = torch.add(z_c * a11, z1 * a12)
        #
        # w21 = torch.var(c_z)
        # w22 = torch.var(z2)
        # a21 = w21 / (w21 + w22)
        # a22 = 1 - a21
        # c_z = torch.add(c_z * a21, z2 * a22)


        return z_c, c_z, attn.squeeze(0)
