import copy
import functools
import os
import time

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        data_mode,

        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        device_id=None,

        save_path
    ):
        self.data_mode = data_mode
        self.device = device_id
        self.save_path = save_path
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            print("self.resume_step : ", self.resume_step)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=self.device
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=self.device
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=self.device
            )
            self.opt.load_state_dict(state_dict)

    # 训练函数入口
    def run_loop(self):
        while (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):

            batch, limited_img_fuck, pre_data, img_path = next(self.data)

            limited_img_fuck = [limited_img_fuck, pre_data]
            # print(img_path)
            # logger.log("img_path : ", img_path)
            # print(limited_img_fuck[0])  # (4, 512, 512)
            # print(batch[0][0][0][0])  # (4, 512, 512)
            # # 展示输入数据
            # limited_img_tmp = np.squeeze(limited_img_fuck.cpu().numpy())
            # batch_tmp = np.squeeze(batch.cpu().numpy())
            # print(limited_img_tmp.shape) # (4, 512, 512)
            # print(batch_tmp.shape) # (4, 512, 512)
            #
            # print(type(batch_tmp[0][0][0]))
            # print(type(limited_img_tmp[0][0][0]))
            #
            # import matplotlib.pyplot as plt
            # # 创建一个 2 行 4 列的子图（每行 4 张图像）
            # fig, axs = plt.subplots(2, 4, figsize=(16, 8))
            #
            # # 绘制 limited_img_tmp 的图像
            # for i in range(4):
            #     axs[0, i].imshow(limited_img_tmp[i], cmap='gray')
            #     axs[0, i].axis('off')  # 隐藏坐标轴
            #     axs[0, i].set_title(f"Limited {i + 1}")
            #
            # # 绘制 batch_tmp 的图像
            # for i in range(4):
            #     axs[1, i].imshow(batch_tmp[i], cmap='gray')
            #     axs[1, i].axis('off')  # 隐藏坐标轴
            #     axs[1, i].set_title(f"Batch {i + 1}")
            #
            # # 调整布局并显示
            # plt.tight_layout()
            # plt.show()
            # return
            # limited_img = th.tensor(limited_img_fuck).to(device=self.device)
            self.run_step(batch, limited_img_fuck)

            if self.step % self.log_interval == 0:
                logger.dumpkvs() 
            if self.step % self.save_interval == 0:
                self.save(self.data_mode)
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save(self.data_mode)

    def run_step(self, batch, limited_img):
        self.forward_backward(batch, limited_img)

        took_step = self.mp_trainer.optimize(self.opt)

        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, limited_img):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):

            micro = batch[i: i + self.microbatch].to(self.device)
            limited_img_tmp = []
            for img in limited_img:
                img[i: i + self.microbatch].to(self.device)
                limited_img_tmp.append(img)
            # limited_img_tmp = limited_img[i: i + self.microbatch].to(self.device)

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                limited_img_tmp,
                t,
                device=self.device
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self, data_mode):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            save_path = self.save_path

            if dist.get_rank() == 0:
                if data_mode == "img":
                    logger.log(f"saving model {rate}...")
                    if not rate:
                        filename = f"mode_{data_mode}_{(self.step+self.resume_step):06d}.pt"
                    else:
                        filename = f"ema_{data_mode}_{rate}_{(self.step+self.resume_step):06d}.pt"

                    with bf.BlobFile(bf.join(save_path, filename), "wb") as f:
                        th.save(state_dict, f)
                        print("Model saved in{save_path}".format(save_path=save_path))
                elif data_mode == "sino":
                    logger.log(f"saving model {rate}...")
                    if not rate:
                        filename = f"model_{data_mode}_{(self.step + self.resume_step):06d}.pt"
                    else:
                        filename = f"ema_{data_mode}_{rate}_{(self.step + self.resume_step):06d}.pt"

                    with bf.BlobFile(bf.join(save_path, filename), "wb") as f:
                        th.save(state_dict, f)
                        print("Model saved in{save_path}".format(save_path=save_path))

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def save_test(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            save_path = self.save_path

            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"Test_model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"Test_ema_{rate}_{(self.step + self.resume_step):06d}.pt"

                with bf.BlobFile(bf.join(save_path, filename), "wb") as f:
                    th.save(state_dict, f)
                    print("Model saved in{save_path}".format(save_path=save_path))

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"Test_opt{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()



class DiffusionDistillation:
    def __init__(
        teacher_model,
        student_model,
        data,
        schedule_sampler=None,
    ):
        pass

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("mode_img_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
