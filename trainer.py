from utils import calc_psnr_and_ssim
from model import Vgg19

import os
import numpy as np
from imageio import imread, imsave
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter

# [0:10,0:10]
def deserialize_roi(roi_string):

    data = roi_string[1:-1]
    data = data.split(',')
    row = data[0].split(':')
    col = data[1].split(':')

    return int(row[0]),int(row[1]),int(col[0]),int(col[1])

def identify_attention(attention_args, roi_string, sr_image, ref_image):

    rowStart, rowEnd, colStart, colEnd = deserialize_roi(roi_string)

    # write a bounding box in the sr image
    sr_image = Image.fromarray(sr_image)
    sr_image_draw = ImageDraw.Draw(sr_image)
    sr_image_draw.rectangle([(colStart,rowStart),(colEnd,rowEnd)], fill=None, outline='red')
    sr_image = np.array(sr_image)

    # get a binary mask of the attention
    attention_pixels = attention_args[rowStart//4 : rowEnd//4 + 1, colStart//4 : colEnd//4 + 1]
    attn_zeros = np.zeros((sr_image.shape[0], sr_image.shape[1], 1), dtype=np.uint8)
    attn_bin_mask = np.copy(attn_zeros)
    attn_bin_mask_ptr = np.ravel(attn_bin_mask)
    for attn_pxl in attention_pixels.reshape(-1):
        attn_bin_mask_ptr[4 * attn_pxl + 2] = 255
    kernel = np.array([[0,1,1,1,0],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [0,1,1,1,0]
                     ], np.uint8)
    attn_bin_mask = cv2.dilate(attn_bin_mask, kernel)
    attn_bin_mask = np.expand_dims(attn_bin_mask, axis=2)

    attn_bin_mask_red = np.concatenate((attn_bin_mask, attn_zeros, attn_zeros), axis=2)
    ref_image = np.maximum(ref_image, attn_bin_mask_red)

    # concat final images
    attn_bin_mask = np.concatenate((attn_bin_mask,attn_bin_mask,attn_bin_mask), axis=2)
    # concat_image = np.concatenate((sr_image, ref_image, attn_bin_mask), axis=1)
    concat_image = np.concatenate((sr_image, ref_image), axis=1)
    return concat_image

class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(self.vgg19, list(range(self.args.num_gpu)))

        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.MainNet.parameters() if
             args.num_gpu==1 else self.model.module.MainNet.parameters()),
             "lr": args.lr_rate
            },
            {"params": filter(lambda p: p.requires_grad, self.model.LTE.parameters() if
             args.num_gpu==1 else self.model.module.LTE.parameters()),
             "lr": args.lr_rate_lte
            }
        ]
        # self.params = [
        #     {"params": filter(lambda p: p.requires_grad, self.model.parameters() if
        #      args.num_gpu==1 else self.model.module.parameters()),
        #      "lr": args.lr_rate
        #     }
        # ]
        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

        self.writer = SummaryWriter(comment=args.dataset)
        self.train_images_visualize = 3
        self.test_images_visualize = 6
        # self.writer.add_hparams(
        #     hparam_dict=
        #     {
        #         "lr_rate": args.lr_rate,
        #         "lr_rate_dis": args.lr_rate_dis,
        #         "lr_rate_lte": args.lr_rate_lte,
        #         "rec_w": args.rec_w,
        #         "per_w": args.per_w,
        #         "tpl_w": args.tpl_w,
        #         "adv_w": args.adv_w
        #     },
        #     metric_dict={})

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            #model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def visualize_reference_images(self, epoch_to_log):
        train_dataloader = self.dataloader['train_no_shuffle']
        for i_batch, train_batch in enumerate(train_dataloader):
            # self.writer.add_images('test/image{}'.format(i_batch), np.uint8((train_batch["LR"] + 1) * 127.5), epoch_to_log)
            self.writer.add_images('train/image{}'.format(i_batch), np.uint8((train_batch["Ref"] + 1) * 127.5), epoch_to_log)
            if i_batch + 1 == self.train_images_visualize:
                break

        test_dataloader = self.dataloader['test']['1']
        for i_batch, test_batch in enumerate(test_dataloader):
            # self.writer.add_images('test/image{}'.format(i_batch), np.uint8((test_batch["LR"] + 1) * 127.5), epoch_to_log)
            self.writer.add_images('test/image{}'.format(i_batch), np.uint8((test_batch["Ref"] + 1) * 127.5), epoch_to_log)
            if i_batch + 1 == self.test_images_visualize:
                break

    def visualize_inference_results(self, current_epoch):

        with torch.no_grad():
            train_dataloader = self.dataloader['train_no_shuffle']
            for i_batch, train_batch in enumerate(train_dataloader):
                train_prepared = self.prepare(train_batch)
                train_sr, _, _, _, _ = self.model(lr=train_prepared['LR'], lrsr=train_prepared['LR_sr'], ref=train_prepared['Ref'], refsr=train_prepared['Ref_sr'])
                train_sr = torch.clamp(train_sr, -1, 1)
                self.writer.add_images('train/image{}'.format(i_batch), np.uint8((train_sr.cpu() + 1) * 127.5), current_epoch)
                if i_batch + 1 == self.train_images_visualize:
                    break

            test_dataloader = self.dataloader['test']['1']
            for i_batch, test_batch in enumerate(test_dataloader):
                test_prepared = self.prepare(test_batch)
                test_sr, _, _, _, _ = self.model(lr=test_prepared['LR'], lrsr=test_prepared['LR_sr'], ref=test_prepared['Ref'], refsr=test_prepared['Ref_sr'])
                test_sr = torch.clamp(test_sr, -1, 1)
                self.writer.add_images('test/image{}'.format(i_batch), np.uint8((test_sr.cpu() + 1) * 127.5), current_epoch)
                if i_batch + 1 == self.test_images_visualize:
                    break

        torch.cuda.empty_cache()

    # epoch is 1-based
    def train(self, current_epoch=0, is_init=False):

        self.model.train()
        if (not is_init):
            self.scheduler.step()

        # all init epochs become negative, the last init epoch becomes 0
        min_epoch_idx = -self.args.num_init_epochs + 1
        current_epoch_absolute = current_epoch + int(is_init) * (min_epoch_idx - 1)

        dataloader = self.dataloader['train']
        logs_per_epoch = (len(dataloader) // self.args.print_every) + 1

        # log only once in the beginning
        if current_epoch_absolute == min_epoch_idx:
            self.logger.info('logs per epoch: {}'.format(logs_per_epoch))
            self.writer.add_text("train/logs_per_epoch", "Logs per epoch: {}".format(logs_per_epoch))
            self.visualize_reference_images(-self.args.num_init_epochs)

        learning_rate = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("train/learning_rate", learning_rate, current_epoch_absolute)

        # cumulative losses
        rec_loss_cum = 0.
        per_loss_cum = 0.
        tpl_loss_cum = 0.
        adv_loss_cum = 0.
        total_loss_cum = 0.

        # instant losses
        rec_loss_curr = 0.
        per_loss_curr = 0.
        tpl_loss_curr = 0.
        adv_loss_curr = 0.

        with tqdm(total=len(dataloader.dataset)) as pbar:
            pbar.set_description("Epoch {}".format(current_epoch_absolute))

            for i_batch, sample_batched in enumerate(dataloader):
                self.optimizer.zero_grad()

                samples_count = sample_batched['LR'].shape[0]
                sample_batched = self.prepare(sample_batched)
                lr = sample_batched['LR']
                lr_sr = sample_batched['LR_sr']
                hr = sample_batched['HR']
                ref = sample_batched['Ref']
                ref_sr = sample_batched['Ref_sr']
                sr, S, T_lv3, T_lv2, T_lv1 = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)

                is_print = ((i_batch + 1) % self.args.print_every == 0) ### flag of print
                iteration_idx = (current_epoch_absolute - 1) * logs_per_epoch + (i_batch + 1) // self.args.print_every

                ### calc loss
                rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
                rec_loss_curr = rec_loss.item()
                rec_loss_cum += rec_loss_curr
                total_loss_cum += rec_loss_curr
                loss = rec_loss
                if is_print:
                    self.writer.add_scalar("train/cum_rec_loss", rec_loss_cum / (i_batch + 1), iteration_idx)

                if (not is_init):
                    if ('per_loss' in self.loss_all):
                        sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                        with torch.no_grad():
                            hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                        per_loss = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                        per_loss_curr = per_loss.item()
                        per_loss_cum += per_loss_curr
                        total_loss_cum += per_loss_curr
                        loss += per_loss
                        if is_print:
                            self.writer.add_scalar("train/cum_per_loss", per_loss_cum / (i_batch + 1), iteration_idx)
                    if ('tpl_loss' in self.loss_all):
                        sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr)
                        tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1,
                            S, T_lv3, T_lv2, T_lv1)
                        tpl_loss_curr = tpl_loss.item()
                        tpl_loss_cum = tpl_loss_curr
                        total_loss_cum += tpl_loss_curr
                        loss += tpl_loss
                        if is_print:
                            self.writer.add_scalar("train/cum_tpl_loss", tpl_loss_cum / (i_batch + 1), iteration_idx)
                    if ('adv_loss' in self.loss_all):
                        adv_loss = self.args.adv_w * self.loss_all['adv_loss'](sr, hr)
                        adv_loss_curr = adv_loss.item()
                        adv_loss_cum += adv_loss_curr
                        total_loss_cum += adv_loss_curr
                        loss += adv_loss
                        if is_print:
                            self.writer.add_scalar("train/cum_adv_loss", adv_loss_cum / (i_batch + 1), iteration_idx)

                if is_print:
                    total_loss_cum_logged = total_loss_cum / (i_batch + 1)
                    self.writer.add_scalar("train/cum_total_loss", total_loss_cum_logged, iteration_idx)
                    pbar.set_postfix(loss=total_loss_cum_logged, refresh=False)

                pbar.update(samples_count)

                loss.backward()
                self.optimizer.step()

                self.writer.flush()

        # log end epoch losses
        last_iteration_idx = current_epoch_absolute * logs_per_epoch
        self.writer.add_scalar("train/cum_rec_loss", rec_loss_cum / len(dataloader), last_iteration_idx)
        if not is_init:
            if ('per_loss' in self.loss_all):
                self.writer.add_scalar("train/cum_per_loss", per_loss_cum / len(dataloader), last_iteration_idx)
            if ('tpl_loss' in self.loss_all):
                self.writer.add_scalar("train/cum_tpl_loss", tpl_loss_cum / len(dataloader), last_iteration_idx)
            if ('adv_loss' in self.loss_all):
                self.writer.add_scalar("train/cum_adv_loss", adv_loss_cum / len(dataloader), last_iteration_idx)
        self.writer.add_scalar("train/cum_total_loss", total_loss_cum / len(dataloader), last_iteration_idx)
        self.visualize_inference_results(current_epoch_absolute)
        self.writer.flush()

        # save model
        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')

        if (self.args.dataset == 'CUFED' or self.args.dataset == 'ffhq'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
                for i_batch, sample_batched in enumerate(self.dataloader['test']['1']):
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']

                    sr, _, _, _, _ = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
                    sr = torch.clamp(sr, -1, 1)
                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'.png'), sr_save)

                    ### calculate psnr and ssim
                    _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())

                    psnr += _psnr
                    ssim += _ssim

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f' %(psnr_ave, ssim_ave))
                self.writer.add_scalar("test/psnr", psnr_ave, current_epoch)
                self.writer.add_scalar("test/ssim", ssim_ave, current_epoch)
                self.writer.flush()
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)'
                    %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))

        self.logger.info('Evaluation over.')

    def test(self):
        self.logger.info('Test process...')
        self.logger.info('lr path:     %s' %(self.args.lr_path))
        self.logger.info('ref path:    %s' %(self.args.ref_path))

        ### LR and LR_sr
        LR = imread(self.args.lr_path)
        h1, w1 = LR.shape[:2]
        LR_sr = np.array(Image.fromarray(LR).resize((w1*4, h1*4), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref = imread(self.args.ref_path)
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        ### to tensor
        LR_t = torch.from_numpy(LR.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        LR_sr_t = torch.from_numpy(LR_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_t = torch.from_numpy(Ref.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_sr_t = torch.from_numpy(Ref_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            if not self.args.attention_visualize:
                sr, _, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t, ref=Ref_t, refsr=Ref_sr_t, return_attention=False)
            else:
                sr, _, S_args, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t, ref=Ref_t, refsr=Ref_sr_t, return_attention=True)
            sr_save = (sr+1.) * 127.5
            sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path))
            imsave(save_path, sr_save)

            if self.args.attention_visualize:
                S_args = S_args.squeeze().cpu().numpy()
                Ref_save = (Ref_t+1) * 127.5
                Ref_save = np.transpose(Ref_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                attn_image = identify_attention(S_args, self.args.attention_roi, sr_save, Ref_save)
                imsave(os.path.join(self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path)[:-4] + "-attention.png"), attn_image)

            self.logger.info('output path: %s' %(save_path))

        self.logger.info('Test over.')