import datetime
import gc
import math
import os
import random
import time
from typing import Any

import numpy as np
import torch.amp
from tqdm import tqdm

import torch
import torch.distributed
from tools.data import build_dataloader
from tools.utils.ckpt import load_ckpt, save_ckpt
from tools.utils.logging import get_logger
from tools.utils.stats import TrainingStats
from tools.utils.utility import AverageMeter

__all__ = ['Trainer', 'prediction_for_metric']

import torch.distributed as dist

rank = int(os.environ.get('RANK', 0))  # torchrun 会提供 RANK


def is_main_process():
    return (not dist.is_available() or not dist.is_initialized() or rank == 0)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def prediction_for_metric(predictions: Any) -> Any:
    """Select refined logits without changing other prediction protocols."""
    if isinstance(predictions, dict) and "refined_logits" in predictions:
        return predictions["refined_logits"]
    return predictions


class Trainer(object):

    def __init__(self, cfg, mode='train', task='rec'):
        self.cfg = cfg.cfg
        if task != 'rec':
            raise ValueError('The MSFE-RSVP repository only supports rec tasks.')
        self.task = 'rec'
        self.report_iterative_ctc_metrics = bool(
            self.cfg['Global'].get('report_iterative_ctc_metrics', False))
        if self.report_iterative_ctc_metrics and task != 'rec':
            raise ValueError(
                'report_iterative_ctc_metrics is only supported for '
                'recognition tasks')
        self.paired_deterministic = bool(
            self.cfg['Global'].get('paired_deterministic', False))
        self.local_rank = (int(os.environ['LOCAL_RANK'])
                           if 'LOCAL_RANK' in os.environ else 0)
        self.set_device(self.cfg['Global']['device'])
        mode = mode.lower()
        assert mode in [
            'train_eval',
            'train',
            'eval',
            'test',
        ], 'mode should be train, eval and test'
        if self.paired_deterministic and 'train' in mode:
            sampling_seed = self.cfg.get('Train', {}).get(
                'dataset', {}).get('sampling_seed')
            if sampling_seed is None:
                raise ValueError(
                    'Global.paired_deterministic=True requires '
                    'Train.dataset.sampling_seed.')
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        use_distributed = (torch.cuda.device_count() > 1 and 'train' in mode
                           and world_size > 1)
        if use_distributed:
            torch.distributed.init_process_group(backend='nccl')
            torch.cuda.set_device(self.device)
            self.cfg['Global']['distributed'] = True
        else:
            self.cfg['Global']['distributed'] = False
            self.local_rank = 0

        self.cfg['Global']['output_dir'] = self.cfg['Global'].get(
            'output_dir', 'output')
        os.makedirs(self.cfg['Global']['output_dir'], exist_ok=True)

        self.writer = None
        if is_main_process(
        ) and self.cfg['Global']['use_tensorboard'] and 'train' in mode:
            import wandb
            from torch.utils.tensorboard import SummaryWriter
            wandb.init(project='demo-sync-tb',
                       name=self.cfg['Global'].get('run_name',
                                                   'log_wandb_openocr'),
                       sync_tensorboard=True)

            self.writer = SummaryWriter(self.cfg['Global']['output_dir'])

        self.logger = get_logger(
            'openrec',
            os.path.join(self.cfg['Global']['output_dir'], 'train.log')
            if 'train' in mode else None,
        )

        cfg.print_cfg(self.logger.info)

        if self.cfg['Global']['device'] == 'gpu' and self.device.type == 'cpu':
            self.logger.info('cuda is not available, auto switch to cpu')

        self.set_random_seed(self.cfg['Global'].get('seed', 48))

        # build data loader
        self.train_dataloader = None
        if 'train' in mode:
            if is_main_process():
                cfg.save(
                    os.path.join(self.cfg['Global']['output_dir'],
                                 'config.yml'), self.cfg)
            self.train_dataloader = build_dataloader(self.cfg,
                                                     'Train',
                                                     self.logger,
                                                     task=task)
            self.logger.info(
                f'train dataloader has {len(self.train_dataloader)} iters')
        self.valid_dataloader = None
        if 'eval' in mode and self.cfg['Eval']:
            try:
                self.valid_dataloader = build_dataloader(self.cfg,
                                                        'Eval',
                                                        self.logger,
                                                        task=task)
                self.logger.info(
                    f'valid dataloader has {len(self.valid_dataloader)} iters')
            except Exception as e:
                self.logger.info(f'valid dataloader build failed: {e}')
                self.valid_dataloader = None

        self._init_rec_model()

        self.logger.info(get_parameter_number(model=self.model))
        self.model = self.model.to(self.device)

        use_sync_bn = self.cfg['Global'].get('use_sync_bn', False)
        if use_sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)
            self.logger.info('convert_sync_batchnorm')
        self._init_train_only_policy()
        self._apply_train_only_freeze()
        self.accumulation_steps = self.cfg['Global'].get(
            'accumulation_steps', 1)
        from openrec.optimizer import build_optimizer
        self.optimizer, self.lr_scheduler = None, None
        epochs = self.cfg['Global']['epoch_num']
        try:
            step_each_epoch = len(self.train_dataloader)
        except TypeError:
            # 针对 IterableDataset 的处理
            step_each_epoch = self.cfg['Global'].get('total_iter_steps', 100000)
        if self.train_dataloader is not None:
            # build optim
            self.optimizer, self.lr_scheduler = build_optimizer(
                self.cfg['Optimizer'],
                self.cfg['LRScheduler'],
                epochs=epochs,
                step_each_epoch=step_each_epoch,
                model=self.model,
            )
        self.grad_clip_val = self.cfg['Global'].get('grad_clip_val', 0)

        self.status = load_ckpt(self.model, self.cfg, self.optimizer,
                                self.lr_scheduler)

        if self.paired_deterministic and 'train' in mode:
            self._reset_training_rng_after_initialization()

        if self.cfg['Global']['distributed']:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, [self.local_rank], find_unused_parameters=False)

        # amp
        self.scaler = (torch.amp.GradScaler() if self.cfg['Global'].get(
            'use_amp', False) else None)

        self.logger.info(
            f'run with torch {torch.__version__} and device {self.device}')

    def _init_rec_model(self):
        from openrec.losses import build_loss as build_rec_loss
        from openrec.metrics import build_metric as build_rec_metric
        from openrec.modeling import build_model as build_rec_model
        from openrec.postprocess import build_post_process as build_rec_post_process

        # build post process
        self.post_process_class = build_rec_post_process(
            self.cfg['PostProcess'], self.cfg['Global'])
        # build model
        # for rec algorithm
        char_num = self.post_process_class.get_character_num()
        self.cfg['Architecture']['Decoder']['out_channels'] = char_num
        self.model = build_rec_model(self.cfg['Architecture'])
        self.use_transformers = False
        # build loss
        self.loss_class = build_rec_loss(self.cfg['Loss'])
        # build metric
        self.eval_class = build_rec_metric(self.cfg['Metric'])
        self.initial_ctc_eval_class = None
        if self.report_iterative_ctc_metrics:
            decoder_config = self.cfg['Architecture'].get('Decoder') or {}
            if not (
                decoder_config.get('name') == 'CTCDecoder'
                and decoder_config.get('use_iterative_ctc', False)
            ):
                raise ValueError(
                    'report_iterative_ctc_metrics requires iterative '
                    'CTCDecoder')
            self.initial_ctc_eval_class = build_rec_metric(
                self.cfg['Metric'])

    def _get_learnable_dfe(self):
        """Return DFE-LW without affecting fixed/legacy model paths."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        encoder = getattr(model, 'encoder', None)
        dfe = getattr(encoder, 'dfe', None)
        if dfe is not None and getattr(dfe, 'fusion_mode', None) == 'learnable':
            return dfe
        return None

    @staticmethod
    def _capture_dfe_lw_norms(module, inputs, output, destination):
        """Capture norms only on an existing low-frequency log iteration."""
        with torch.no_grad():
            source = inputs[0].detach().float()
            result = output.detach().float()
            projected = module.input_projection(source)
            batch, tokens, _ = projected.shape
            f0 = projected.reshape(batch * tokens, 1,
                                   module.embedding_height,
                                   module.embedding_width)
            f1, f2, f3 = f0.flip(2), f0.flip(3), f0.flip((2, 3))
            weights = module.get_branch_weights().detach().float()
            enhance = weights[0] * f1 + weights[1] * f2 + weights[2] * f3
            delta = result - source
            destination.update({
                'dfe_input_norm': float(source.norm()),
                'dfe_enhance_norm': float(enhance.norm()),
                'dfe_output_norm': float(result.norm()),
                'dfe_delta_norm': float(delta.norm()),
                'dfe_delta_ratio': float(
                    delta.norm() / (source.norm() + 1.0e-6)),
            })

    @staticmethod
    def _current_dfe_lw_weights(module):
        with torch.no_grad():
            weights = module.get_branch_weights().detach().float()
            entropy = -(weights * torch.log(weights + 1.0e-6)).sum()
            entropy = entropy / math.log(3.0)
            return {
                'dfe_weight_f1': float(weights[0]),
                'dfe_weight_f2': float(weights[1]),
                'dfe_weight_f3': float(weights[2]),
                'dfe_weight_entropy': float(entropy),
            }

    def _init_train_only_policy(self):
        cfg = self.cfg.get('TrainOnly', {}) or {}
        self.train_only_enabled = bool(cfg.get('enabled', False))
        keywords = cfg.get('trainable_keywords', None)
        if isinstance(keywords, str):
            keywords = [item.strip() for item in keywords.split(',')]
        self.train_only_keywords = [
            str(item) for item in (keywords or []) if str(item)
        ]
        self.train_only_name = str(cfg.get('name', 'TrainOnly'))
        self.log_trainable_parameter_names = bool(
            cfg.get('log_trainable_parameter_names', False))
        if self.train_only_enabled and not self.train_only_keywords:
            raise ValueError(
                'TrainOnly.enabled=True requires non-empty trainable_keywords.')

    def _apply_train_only_freeze(self):
        if not self.train_only_enabled:
            return
        total_params = 0
        trainable_params = 0
        trainable_tensors = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            param.requires_grad = any(
                keyword in name for keyword in self.train_only_keywords)
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_tensors += 1
        if trainable_tensors == 0:
            raise ValueError(
                f'{self.train_only_name} matched no parameters: '
                f'{self.train_only_keywords}')
        self.logger.info(
            f'{self.train_only_name} enabled: '
            f'keywords={self.train_only_keywords}, '
            f'trainable_tensors={trainable_tensors}, '
            f'trainable_params={trainable_params}/{total_params}')
        if self.log_trainable_parameter_names:
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self.logger.info(f'trainable parameter: {name}')

    def _forward_eval_predictions(self, images, data):
        """Return refined predictions and optional raw initial CTC logits."""
        if not self.report_iterative_ctc_metrics:
            return self.model(images, data=data), None

        recognizer = (
            self.model.module
            if isinstance(
                self.model,
                torch.nn.parallel.DistributedDataParallel,
            ) else self.model
        )
        decoder = getattr(recognizer, 'decoder', None)
        classifier = getattr(decoder, 'fc', None)
        if classifier is None:
            raise RuntimeError(
                'iterative CTC metric reporting requires shared decoder.fc')

        classifier_outputs = []

        def capture_classifier_output(module, inputs, output):
            del module, inputs
            classifier_outputs.append(output.detach())

        handle = classifier.register_forward_hook(capture_classifier_output)
        try:
            predictions = self.model(images, data=data)
        finally:
            handle.remove()
        if len(classifier_outputs) != 2:
            raise RuntimeError(
                'iterative CTC metric reporting expected shared decoder.fc '
                f'to run twice, got {len(classifier_outputs)} calls')
        return predictions, classifier_outputs[0]

    def _apply_train_only_module_modes(self):
        if not self.train_only_enabled:
            return
        frozen_types = (
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.Dropout,
            torch.nn.Dropout2d,
            torch.nn.Dropout3d,
        )
        for submodule in self.model.modules():
            if not isinstance(submodule, frozen_types):
                continue
            if not any(param.requires_grad
                       for param in submodule.parameters(recurse=False)):
                submodule.eval()

    def _set_component_head_warmup_epoch(self, epoch):
        model = (
            self.model.module
            if isinstance(
                self.model,
                torch.nn.parallel.DistributedDataParallel,
            ) else self.model
        )
        decoder = getattr(model, 'decoder', None)
        setter = getattr(decoder, 'set_component_training_epoch', None)
        if setter is None:
            return False
        active = bool(setter(epoch))
        warmup_epochs = int(
            getattr(decoder, 'component_head_warmup_epochs', 0))
        if warmup_epochs > 0 and (
                epoch <= warmup_epochs + 1 or epoch == 1):
            self.logger.info(
                'component head warmup: '
                f'epoch={epoch}, warmup_epochs={warmup_epochs}, '
                f'feature_detach={active}')
        return active

    def load_params(self, params):
        self.model.load_state_dict(params)

    def set_random_seed(self, seed):
        seed = int(seed)
        if self.paired_deterministic:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            # CUDA CTC backward has no deterministic implementation in
            # PyTorch 2.8. Keep deterministic implementations enabled where
            # available, but allow the shared U0R/U6R CTC path to proceed.
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(seed)  # 为CPU设置随机种子
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = not self.paired_deterministic
            torch.backends.cudnn.deterministic = self.paired_deterministic
            torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        random.seed(seed)
        np.random.seed(seed)

    def _reset_training_rng_after_initialization(self):
        seed = int(self.cfg['Global'].get('seed', 48))
        self.set_random_seed(seed)
        self.logger.info(
            'paired deterministic RNG reset after model, optimizer, and '
            f'checkpoint initialization: seed={seed}')

    def set_device(self, device):
        if device == 'gpu' and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.local_rank}')
        else:
            device = torch.device('cpu')
        self.device = device

    def _uses_train_time_decoding_metric(self):
        decoder_name = self.cfg.get('Architecture', {}).get('Decoder',
                                                            {}).get('name')
        return decoder_name in {
            'ASTERDecoder',
            'SARDecoder',
            'PARSeqDecoder',
            'NRTRDecoder',
            'ABINetDecoder',
        }

    def _is_better_metric(self, cur_metric):
        main_indicator = self.eval_class.main_indicator
        cur_main = cur_metric.get(main_indicator)
        best_main = self.best_metric.get(main_indicator)
        if cur_main is None:
            return False
        if best_main is None or 'best_epoch' not in self.best_metric:
            return True

        eps = 1e-12
        if cur_main > best_main + eps:
            return True
        if cur_main < best_main - eps:
            return False

        # When the main metric ties, keep the checkpoint with better
        # character-level quality instead of repeatedly overwriting best.pth.
        tie_breaker = 'norm_edit_dis'
        if main_indicator.startswith('gtc_') and 'gtc_norm_edit_dis' in cur_metric:
            tie_breaker = 'gtc_norm_edit_dis'
        if tie_breaker != main_indicator and tie_breaker in cur_metric:
            best_tie = self.best_metric.get(tie_breaker)
            if best_tie is None:
                return True
            return cur_metric[tie_breaker] > best_tie + eps
        return False

    def train(self):
        cal_metric_during_train = self.cfg['Global'].get(
            'cal_metric_during_train', False)
        log_smooth_window = self.cfg['Global']['log_smooth_window']
        epoch_num = self.cfg['Global']['epoch_num']
        print_batch_step = self.cfg['Global']['print_batch_step']
        eval_epoch_step = self.cfg['Global'].get('eval_epoch_step', 1)

        start_eval_epoch = 0
        if self.valid_dataloader is not None:
            if type(eval_epoch_step) == list and len(eval_epoch_step) >= 2:
                start_eval_epoch = eval_epoch_step[0]
                eval_epoch_step = eval_epoch_step[1]
                if len(self.valid_dataloader) == 0:
                    start_eval_epoch = 1e111
                    self.logger.info(
                        'No Images in eval dataset, evaluation during training will be disabled'
                    )
                self.logger.info(
                    f'During the training process, after the {start_eval_epoch}th epoch, '
                    f'an evaluation is run every {eval_epoch_step} epoch')
        else:
            start_eval_epoch = 1e111

        eval_batch_step = self.cfg['Global']['eval_batch_step']

        global_step = self.status.get('global_step', 0)

        start_eval_step = 0
        if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
            start_eval_step = eval_batch_step[0]
            eval_batch_step = eval_batch_step[1]
            if self.valid_dataloader is not None and len(self.valid_dataloader) == 0:
                self.logger.info(
                    'No Images in eval dataset, evaluation during training '
                    'will be disabled')
                start_eval_step = 1e111
            self.logger.info(
                'During the training process, after the {}th iteration, '
                'an evaluation is run every {} iterations'.format(
                    start_eval_step, eval_batch_step))

        save_epoch_step = self.cfg['Global'].get('save_epoch_step', [0, 1])
        start_save_epoch = save_epoch_step[0]
        save_epoch_step = save_epoch_step[1]

        start_epoch = self.status.get('epoch', 1)
        self.best_metric = self.status.get('metrics', {})
        if self.eval_class.main_indicator not in self.best_metric:
            self.best_metric[self.eval_class.main_indicator] = 0
        self.secondary_best_indicator = self.cfg['Global'].get(
            'secondary_best_indicator')
        self.best_secondary_metric = {}
        train_stats = TrainingStats(log_smooth_window, ['lr'])
        self.model.train()
        self._apply_train_only_module_modes()

        if cal_metric_during_train and self._uses_train_time_decoding_metric(
        ) and is_main_process():
            self.logger.info(
                'For autoregressive/language-model recognizers, training '
                'metrics are computed from train-time decoder outputs and may '
                'use teacher forcing or masked targets. Use eval metrics for '
                'fair model comparison.')

        total_samples = 0
        train_reader_cost = 0.0
        train_batch_cost = 0.0
        reader_start = time.time()
        eta_meter = AverageMeter()
        save_iter_step = self.cfg['Global'].get('save_iter_step',
                                                [10e10, 2000])
        start_save_iter = save_iter_step[0]
        save_iter_step = save_iter_step[1]

        if self.cfg['Global'].get('resume_from_iter',
                                  False):  # for unirec resume training
            if self.cfg['Global']['checkpoints'] is None:
                raise ValueError(
                    'resume_from_iter is True, but checkpoints is None')
            start_epoch = start_epoch - 1
            self.resume_iter = global_step
            iter_model_file_name = os.path.basename(
                self.cfg['Global']['checkpoints'])
            last_whole_epoch_global_step = iter_model_file_name.split('_')[1]
            self.cfg['Train']['sampler'][
                'resume_iter'] = self.resume_iter - last_whole_epoch_global_step

        last_whole_epoch_global_step = 0
        for epoch in range(start_epoch, epoch_num + 1):
            self._set_component_head_warmup_epoch(epoch)
            set_loss_epoch = getattr(self.loss_class, 'set_epoch', None)
            if set_loss_epoch is not None:
                set_loss_epoch(epoch)
            if not self.cfg['Global'].get('resume_from_iter',
                                          False):  # for unirec resume training
                if 'sampler' in self.cfg['Train']:
                    self.cfg['Train']['sampler']['resume_iter'] = 0
            if hasattr(self.train_dataloader, 'dataset') and self.train_dataloader.dataset is not None:
                if self.train_dataloader.dataset.need_reset and epoch > 1:
                    if hasattr(self.train_dataloader.dataset, 'close'):
                        self.train_dataloader.dataset.close()
                    old_train_dataloader = self.train_dataloader
                    self.train_dataloader = None
                    del old_train_dataloader
                    gc.collect()
                    self.train_dataloader = build_dataloader(self.cfg,
                                                            'Train',
                                                            self.logger,
                                                            epoch=epoch,
                                                            task=self.task)

            for idx, batch in enumerate(self.train_dataloader):
                if self.cfg['Global'].get('resume_from_iter',
                                          False):  # for unirec resume training
                    if global_step != self.resume_iter:
                        global_step += 1
                        if is_main_process(
                        ) and global_step % print_batch_step == 0:
                            self.logger.info(
                                f'skip iter {global_step}, resume from iter {self.resume_iter}'
                            )
                        continue
                    else:
                        global_step += 1
                        self.cfg['Global']['resume_from_iter'] = False
                        self.logger.info(
                            f'resume from iter {self.resume_iter}, start training from iter {global_step}'
                        )
                        continue

                batch_tensor = [t.to(self.device) for t in batch]
                batch_numpy = [t.numpy() for t in batch]
                train_reader_cost += time.time() - reader_start
                next_global_step = global_step + 1
                should_log_iteration = is_main_process() and (
                    (next_global_step > 0 and
                     next_global_step % print_batch_step == 0) or
                    (idx >= len(self.train_dataloader) - 1))
                dfe_lw_module = self._get_learnable_dfe()
                dfe_lw_diagnostics = {}
                dfe_lw_hook = None
                if dfe_lw_module is not None and should_log_iteration:
                    def capture_dfe_lw(module, inputs, output):
                        self._capture_dfe_lw_norms(
                            module, inputs, output, dfe_lw_diagnostics)

                    dfe_lw_hook = dfe_lw_module.register_forward_hook(
                        capture_dfe_lw)
                # use amp
                if self.scaler:
                    with torch.amp.autocast(device_type=self.device.type,
                                            dtype=torch.bfloat16):
                        if self.use_transformers:
                            inputs = {
                                'pixel_values': batch_tensor[0],
                                'input_ids': None,
                                'attention_mask': None,
                                'labels': batch_tensor[1],
                                'length': batch_tensor[2]
                            }
                            preds = self.model(**inputs)
                        else:
                            preds = self.model(batch_tensor[0],
                                               data=batch_tensor[1:])
                        loss = self.loss_class(preds, batch_tensor)
                        loss['loss'] = loss['loss'] / self.accumulation_steps
                    self.scaler.scale(loss['loss']).backward()
                    if (global_step + 1) % self.accumulation_steps == 0:
                        if self.grad_clip_val > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.grad_clip_val)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    preds = self.model(batch_tensor[0], data=batch_tensor[1:])
                    loss = self.loss_class(preds, batch_tensor)
                    avg_loss = loss['loss']
                    avg_loss.backward()
                    if self.grad_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_clip_val)
                    self.optimizer.step()
                if dfe_lw_hook is not None:
                    dfe_lw_hook.remove()

                if cal_metric_during_train:  # only rec and cls need
                    metric_preds = prediction_for_metric(preds)
                    post_result = self.post_process_class(metric_preds,
                                                          batch_numpy,
                                                          training=True)
                    self.eval_class(post_result, batch_numpy, training=True)
                    metric = self.eval_class.get_metric()
                    train_stats.update(metric)

                train_batch_time = time.time() - reader_start
                train_batch_cost += train_batch_time
                eta_meter.update(train_batch_time)
                global_step += 1
                total_samples += len(batch[0])

                try:
                    self.lr_scheduler.step()
                except Exception as e:
                    self.logger.info(
                        f'lr_scheduler step error, {e}, please check your config'
                    )

                loss['loss'] = loss['loss'] * self.accumulation_steps
                # logger
                stats = {
                    k: float(v)
                    if v.shape == [] else v.detach().cpu().numpy().mean()
                    for k, v in loss.items()
                }
                stats['lr'] = self.lr_scheduler.get_last_lr()[0]
                train_stats.update(stats)

                if self.writer is not None:
                    for k, v in train_stats.get().items():
                        self.writer.add_scalar(f'TRAIN/{k}', v, global_step)

                if is_main_process() and (
                    (global_step > 0 and global_step % print_batch_step == 0)
                        or (idx >= len(self.train_dataloader) - 1)):
                    logs = train_stats.log()

                    dfe_lw_log = ''
                    if dfe_lw_module is not None:
                        dfe_lw_diagnostics.update(
                            self._current_dfe_lw_weights(dfe_lw_module))
                        dfe_lw_log = ', ' + ', '.join(
                            f'{key}: {value:.8f}'
                            for key, value in dfe_lw_diagnostics.items())

                    eta_sec = (
                        (epoch_num + 1 - epoch) * len(self.train_dataloader) -
                        idx - 1) * eta_meter.avg
                    eta_sec_format = str(
                        datetime.timedelta(seconds=int(eta_sec)))
                    strs = (
                        f'epoch: [{epoch}/{epoch_num}], global_step: {global_step}, {logs}{dfe_lw_log}, '
                        f'avg_reader_cost: {train_reader_cost / print_batch_step:.5f} s, '
                        f'avg_batch_cost: {train_batch_cost / print_batch_step:.5f} s, '
                        f'avg_samples: {total_samples / print_batch_step}, '
                        f'ips: {total_samples / train_batch_cost:.5f} samples/s, '
                        f'eta: {eta_sec_format}')
                    self.logger.info(strs)
                    total_samples = 0
                    train_reader_cost = 0.0
                    train_batch_cost = 0.0
                reader_start = time.time()
                # eval iter step
                if self.valid_dataloader is not None and is_main_process() and (global_step > start_eval_step and
                                          (global_step - start_eval_step) %
                                          eval_batch_step == 0):
                    self.eval_step(global_step, epoch)
                # save iter step
                if is_main_process(
                ) and global_step > start_save_iter and global_step % save_iter_step == 0:
                    save_ckpt(
                        self.model,
                        self.cfg,
                        self.optimizer,
                        self.lr_scheduler,
                        epoch,
                        global_step,
                        self.best_metric,
                        is_best=False,
                        prefix=
                        f'iter_{last_whole_epoch_global_step}_{global_step}')

            # eval epoch step
            if self.valid_dataloader is not None and is_main_process() and epoch > start_eval_epoch and (
                    epoch - start_eval_epoch) % eval_epoch_step == 0:
                self.eval_step(global_step, epoch)

            if is_main_process():
                save_ckpt(self.model,
                          self.cfg,
                          self.optimizer,
                          self.lr_scheduler,
                          epoch,
                          global_step,
                          self.best_metric,
                          is_best=False,
                          prefix=None)
                if epoch > start_save_epoch and (
                        epoch - start_save_epoch) % save_epoch_step == 0:
                    save_ckpt(self.model,
                              self.cfg,
                              self.optimizer,
                              self.lr_scheduler,
                              epoch,
                              global_step,
                              self.best_metric,
                              is_best=False,
                              prefix='epoch_' + str(epoch))
            last_whole_epoch_global_step = global_step
        best_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in self.best_metric.items()])}"
        self.logger.info(best_str)
        if self.writer is not None:
            self.writer.close()
        if self.cfg['Global']['distributed']:
            torch.distributed.barrier()

    def eval_step(self, global_step, epoch):
        cur_metric = self.eval()
        cur_metric_str = f"cur metric, {', '.join(['{}: {}'.format(k, v) for k, v in cur_metric.items()])}"
        self.logger.info(cur_metric_str)

        # logger metric
        if self.writer is not None:
            for k, v in cur_metric.items():
                if isinstance(v, (float, int)):
                    self.writer.add_scalar(f'EVAL/{k}', cur_metric[k],
                                           global_step)

        secondary = self.secondary_best_indicator
        if secondary and secondary in cur_metric:
            best_secondary = self.best_secondary_metric.get(secondary)
            if best_secondary is None or cur_metric[secondary] > best_secondary:
                self.best_secondary_metric = dict(cur_metric)
                self.best_secondary_metric['best_epoch'] = epoch
                save_ckpt(
                    self.model,
                    self.cfg,
                    self.optimizer,
                    self.lr_scheduler,
                    epoch,
                    global_step,
                    self.best_secondary_metric,
                    is_best=True,
                    prefix=f'best_{secondary}',
                )
                self.logger.info(
                    'best %s: %s, best_epoch: %s',
                    secondary,
                    self.best_secondary_metric[secondary],
                    epoch,
                )

        if self._is_better_metric(cur_metric):
            self.best_metric.update(cur_metric)
            self.best_metric['best_epoch'] = epoch

            if self.writer is not None:
                self.writer.add_scalar(
                    f'EVAL/best_{self.eval_class.main_indicator}',
                    self.best_metric[self.eval_class.main_indicator],
                    global_step,
                )

            save_ckpt(self.model,
                      self.cfg,
                      self.optimizer,
                      self.lr_scheduler,
                      epoch,
                      global_step,
                      self.best_metric,
                      is_best=True,
                      prefix=None)
        best_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in self.best_metric.items()])}"
        self.logger.info(best_str)

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            total_frame = 0.0
            total_time = 0.0
            pbar = tqdm(
                total=len(self.valid_dataloader),
                desc='eval model:',
                position=0,
                leave=True,
            )
            sum_images = 0
            for idx, batch in enumerate(self.valid_dataloader):
                batch_tensor = [t.to(self.device) for t in batch]
                batch_numpy = [t.numpy() for t in batch]
                start = time.time()
                if self.scaler:
                    with torch.cuda.amp.autocast(
                            enabled=self.device.type == 'cuda'):
                        preds, initial_logits = (
                            self._forward_eval_predictions(
                                batch_tensor[0],
                                batch_tensor[1:],
                            )
                        )
                else:
                    preds, initial_logits = self._forward_eval_predictions(
                        batch_tensor[0],
                        batch_tensor[1:],
                    )

                total_time += time.time() - start
                # Obtain usable results from post-processing methods
                # Evaluate the results of the current batch
                post_result = self.post_process_class(preds, batch_numpy)
                self.eval_class(post_result, batch_numpy)
                if initial_logits is not None:
                    initial_probabilities = torch.softmax(
                        initial_logits,
                        dim=-1,
                    )
                    initial_post_result = self.post_process_class(
                        initial_probabilities,
                        batch_numpy,
                    )
                    self.initial_ctc_eval_class(
                        initial_post_result,
                        batch_numpy,
                    )

                pbar.update(1)
                total_frame += len(batch[0])
                sum_images += 1
            # Get final metric，eg. acc or hmean
            metric = self.eval_class.get_metric()
            if self.initial_ctc_eval_class is not None:
                initial_metric = self.initial_ctc_eval_class.get_metric()
                metric['p0_acc'] = initial_metric['acc']
                metric['p1_acc'] = metric['acc']

        pbar.close()
        self.model.train()
        self._apply_train_only_module_modes()
        metric['fps'] = total_frame / total_time
        return metric

    def test_dataloader(self):
        starttime = time.time()
        count = 0
        try:
            for data in self.train_dataloader:
                count += 1
                if count % 1 == 0:
                    batch_time = time.time() - starttime
                    starttime = time.time()
                    self.logger.info(
                        f'reader: {count}, {data[0].shape}, {batch_time}')
        except:
            import traceback

            self.logger.info(traceback.format_exc())
        self.logger.info(f'finish reader: {count}, Success!')
