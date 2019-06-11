import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from opt import opt
from data import Data
from network import MGN
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from utils.logger import setup_logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)
# logger.basicConfig(level=logger.INFO,
#                     format='%(asctime)s %(levelname)s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     filename=os.path.join(opt.output_dir, 'log.txt')
#                    )
# logger.getLogger().addHandler(logger.StreamHandler(sys.stdout))
logger = setup_logger("reid_MGN", opt.output_dir, 0)

writer = SummaryWriter(os.path.join(opt.output_dir, 'runs/'))
global global_step
global_step = 0
class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to('cuda')
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):
        global global_step

        self.scheduler.step()

        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            writer.add_scalar('batch/loss', loss.cpu().item(), global_step)
            global_step += 1
            self.optimizer.step()

    def evaluate(self):

        self.model.eval()

        logger.info('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()

        if opt.dataset == 'csm':
            gf = qf.copy()
        else:
            gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        logger.info('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        logger.info('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        logger.info('extract features, this may take a few minutes')
        query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))

        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small

        # # Remove junk images
        # junk_index = np.argwhere(gallery_label == -1)
        # mask = np.in1d(index, junk_index, invert=True)
        # index = index[mask]

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        logger.info('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[index[i]]
            logger.info(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig(os.path.join(opt.output_dir, "show.png"))
        logger.info('result saved to show.png')


if __name__ == '__main__':


    data = Data(name=opt.dataset)  # 'csm' or 'market1501'
    model = MGN()
    loss = Loss()
    main = Main(model, loss, data)

    if opt.mode == 'train':

        for epoch in range(1, opt.epoch + 1):
            logger.info('\nepoch {}'.format(epoch))
            main.train()
            # if epoch % 50 == 0:
            if epoch % 10 == 0:
                logger.info('\nstart evaluate')
                main.evaluate()
                save_path = os.path.join(opt.output_dir, 'model_{}.pt'.format(epoch))
                torch.save(model.state_dict(), save_path)
                logger.info('model saved in {}'.format(save_path))
    if opt.mode == 'evaluate':
        logger.info('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()

    if opt.mode == 'vis':
        logger.info('visualize')
        model.load_state_dict(torch.load(opt.weight))
        main.vis()
