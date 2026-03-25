import torch
from Models.MyBackbone import *
from Models.DataLoaderSkin import *
import torchutils as tu
from Models.loss import *
import torch.optim as optim
from Models.Util import *
from torch.autograd import Variable
from pytorch_lightning import seed_everything
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from Models.EMATeacher import *
from torch_ema import ExponentialMovingAverage
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class SSL_FM(nn.Module):
    def __init__(self, opt):
        super(SSL_FM, self).__init__()
        seed_everything(opt.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        opt = argparse.Namespace(**vars(opt))
        self.opt = opt
        self.loader_L, self.loader_UL, self.loader_Test = getDataLoaders(self.opt.dataset, self.opt.itr, opt.seed,
                                                                         opt.pL, opt.addValDataWithTrain, self.opt.bs_l,
                                                                         self.opt.bs_u)
        self.labeled_iter = iter(self.loader_L)
        if self.loader_UL is not None:
            self.unlabeled_iter = iter(self.loader_UL)

        y = torch.from_numpy(self.loader_L.dataset.lblArr).cuda(self.opt.gpuid)
        self.uniqueLbls = torch.unique(y)
        self.opt.n_class = len(self.uniqueLbls)

        self.cw_L = None
        self.cw_UL = None
        if self.opt.use_cw:
            self.cw_L = calWeights_GPU(y, self.uniqueLbls, self.opt.gpuid)

        self.criterion_class = CELoss(self.uniqueLbls, self.opt.gpuid)
        self.sm = nn.Softmax(dim=1)
        self.trainingPara = getTrainingPara(self.opt.dataset, self.opt.type)

        self.student = self.loadModel(self.opt.projectionDim)
        self.teacher = EMATeacher(self.student, self.opt.ema_model, self.trainingPara.total_iterations)
        self.setOptimizer(self.opt.lr, self.trainingPara.total_iterations, nwarmup=300)

    def setOptimizer(self, lr, total_iterations, nwarmup):
        if self.opt.optimizer == 'ADAM':
            self.optimizer = optim.Adam(self.student.parameters(), lr=lr, weight_decay=5e-4)
        elif self.opt.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.student.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=nwarmup,
                                                         num_training_steps=total_iterations)

    def loadModel(self, projectDim):
        net = MyBackbone(self.opt.modelName, self.opt.pretrain, self.opt.n_class, projectDim)
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Total trainable parameters = ', (pytorch_total_params // 1000000))
        return net.cuda(self.opt.gpuid)

    def getNextBatch_L(self):
        try:
            Iw, Is_1, lbls, idx = next(self.labeled_iter)
            if len(lbls) <= 1:
                self.labeled_iter = iter(self.loader_L)
                Iw, Is_1, lbls, idx = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.loader_L)
            Iw, Is_1, lbls, idx = next(self.labeled_iter)
        lbls = Variable(lbls.cuda(self.opt.gpuid))
        Iw = Variable(Iw.cuda(self.opt.gpuid), requires_grad=False)
        Is_1 = Variable(Is_1.cuda(self.opt.gpuid), requires_grad=False)
        return Iw, Is_1, lbls, idx

    def getNextBatch_UL(self):
        try:
            Iw, Is_1, lbls, idx = next(self.unlabeled_iter)
            if len(lbls) <= 1:
                self.unlabeled_iter = iter(self.loader_UL)
                Iw, Is_1, lbls, idx = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.loader_UL)
            Iw, Is_1, lbls, idx = next(self.unlabeled_iter)
        lbls = Variable(lbls.cuda(self.opt.gpuid))
        Iw = Variable(Iw.cuda(self.opt.gpuid), requires_grad=False)
        Is_1 = Variable(Is_1.cuda(self.opt.gpuid), requires_grad=False)
        return Iw, Is_1, lbls, idx

    def test_oneImg(self, I):
        self.student.eval()
        with torch.no_grad():
            logits, _, _ = self.teacher(I)
        return logits

    def test(self):
        self.teacher.eval()
        gt_all, logits_all = [], []
        for i, (I, lbls) in enumerate(self.loader_Test):
            I = Variable(I.cuda(self.opt.gpuid))
            lbls = Variable(lbls.cuda(self.opt.gpuid))
            logits = self.test_oneImg(I)

            logits_all.extend(logits)
            gt_all.extend(lbls)
        gt_all = torch.stack(gt_all, 0)
        logits_all = torch.stack(logits_all, 0)
        reClass, desc = getScores_new(gt_all, logits_all)
        return reClass, desc

    def getLblsAndMask(self, prob):
        prob, pred = torch.max(prob, dim=1)
        mask = prob.ge(self.opt.thr)
        pred = Variable(pred, requires_grad=False)
        mask = Variable(mask, requires_grad=False)
        return pred, mask.float(), prob

    def train_sup(self, epoch):
        loss_tot, logits_all, gt_all = 0, [], []

        self.student.train()
        for i in range(self.trainingPara.iterPerEpoch):
            _, Is, lbls, idx_l = self.getNextBatch_L()

            Is = Variable(Is.cuda(self.opt.gpuid), requires_grad=False)
            lbls = Variable(lbls.cuda(self.opt.gpuid), requires_grad=False)

            logits, _, _ = self.student(Is)
            loss =  self.criterion_class(logits, lbls, self.cw_L)

            self.optimizer.zero_grad()
            loss.backward()
            loss_tot += loss.item()
            self.optimizer.step()
            self.scheduler.step()
            self.teacher.update()

            gt_all.extend(lbls)
            logits_all.extend(logits.data.detach())

        gt_all = torch.stack(gt_all, 0)
        logits_all = torch.stack(logits_all, 0)

        reClass, desc = getScores2(gt_all, logits_all)
        return loss_tot, reClass, desc

    def trainSemiSup_FixMatch(self, epoch):
        loss_tot = 0
        Y_l_all, Y_u_all, Y_u_p_all, m_all = [], [], [], []
        self.student.train()
        for i in range(self.trainingPara.iterPerEpoch):
            Il_w, Il_s1, y_l, idx_l = self.getNextBatch_L()
            Iu_w, Iu_s1, y_u, idx_un = self.getNextBatch_UL()
            bs_l, bs_u = Il_w.shape[0], Iu_w.shape[0]

            loss = 0
            logits, _, _ = self.student(torch.cat((Il_s1, Iu_w, Iu_s1)))
            log_w, log_s = logits[bs_l:, :].chunk(2)
            pseudo_label, mask, _ = self.getLblsAndMask(log_w)
            loss = self.criterion_class(logits[:bs_l, :], y_l, self.cw_L)

            if mask.sum() > 0:
                loss += self.criterion_class(log_s, pseudo_label, self.cw_UL, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss_tot += loss.item()
            Y_l_all.extend(y_l)
            Y_u_all.extend(y_u)
            Y_u_p_all.extend(pseudo_label)
            m_all.extend(mask)

        Y_l_all = torch.stack(Y_l_all, 0)
        Y_u_all = torch.stack(Y_u_all, 0)

        accPseudoLbls, bacc, ns = 0, 0, 0
        if self.opt.w_pl > 0:
            Y_u_p_all = torch.stack(Y_u_p_all, 0)
            m_all = torch.stack(m_all, 0)

            if torch.sum(m_all) > 0:
                idx = m_all == 1
                selected_gt = Y_u_all[idx].view(-1, 1).data.cpu().numpy()
                selected_pl = Y_u_p_all[idx].view(-1, 1).data.cpu().numpy()
                accPseudoLbls = accuracy_score(selected_gt, selected_pl)
                bacc = balanced_accuracy_score(selected_gt, selected_pl)
                ns = torch.sum(m_all).cpu().numpy()
        return loss_tot, ns, accPseudoLbls, bacc

    def printStatPL(self, y_all, y_s, pl_s):
        unique_lbls = torch.unique(y_all)
        print('lbl \t tot \t selected \t acc')
        for lbl in unique_lbls:
            idx = pl_s == lbl
            if idx.float().sum() > 0:
                tot = torch.sum(y_all == lbl).data.item()
                selected = torch.sum(idx).data.item()
                acc = accuracy_score(y_s[idx].data.cpu().numpy(), pl_s[idx].data.cpu().numpy()) * 100
                print('%1d\t%5d\t%5d\t%2.2f%%' % (lbl, tot, selected, acc))
        acc = accuracy_score(y_s.data.cpu().numpy(), pl_s.data.cpu().numpy()) * 100
        print('\t%5d\t%5d\t%2.2f%%' % (len(y_all), len(y_s), acc))
        print()

    def iterate(self):
        print('n_epochs: ', self.trainingPara.n_epochs)
        print('iterPerEpoch: ', self.trainingPara.iterPerEpoch)

        results = []
        for epoch in range(self.trainingPara.n_epochs):
            accPseudoLbls, bacc, n_un, accTr, thr = 0, 0, 0, 0, self.opt.thr
            lr = tu.get_lr(self.optimizer)

            if epoch < 5:
                lossTr, reClass, desc = self.train_sup(epoch)
            else:
                lossTr, n_un, accPseudoLbls, bacc = self.trainSemiSup_FixMatch(epoch)
            reClass, desc = self.test()

            print('%s %3d %.6f  |' % (self.opt.type, epoch, lr), end='')
            print(' %.4f %5d %3.2f%% %3.2f%%\t| %5.3f %3.2f%%\t||'
                  % (thr, n_un, accPseudoLbls * 100, bacc * 100, lossTr, accTr), end='')
            print(' %3.2f  %3.2f %3.2f  %3.2f\t'
                  % (reClass[0], reClass[1], reClass[2], reClass[3]), end='|')
            print()
            results.append(reClass)

        print()
        remean = np.mean(results[-5:], axis=0)
        for d in remean:
            print(' %3.2f  ' % (d), end='\t')
        print()
        # desc.extend(desc)
        return remean, desc

