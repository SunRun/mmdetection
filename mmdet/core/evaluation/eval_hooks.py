import os
import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader

from collections import defaultdict


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(
        self,
        dataloader,
        interval=1,
        save_optimizer=True,
        checkpoint_metrics=["bbox_mAP", "segm_mAP"],
        out_dir=None,
        **eval_kwargs
    ):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                "dataloader must be a pytorch DataLoader, but got {}".format(
                    type(dataloader)
                )
            )
        self.dataloader = dataloader
        self.interval = interval
        self.save_optimizer = save_optimizer
        # Best model metrics (value, epoch)
        self.best_model_metrics = defaultdict(lambda: (0, 0))
        self.checkpoint_metrics = checkpoint_metrics
        self.out_dir = out_dir
        # best_metric_value_epoch
        self.filename_tmpl = "best_{}_{}_{{}}"
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import single_gpu_test

        results = single_gpu_test(runner.model, self.dataloader, show=False)
        eval_res = self.evaluate(runner, results)
        if self.checkpoint_metrics is not None:
            self.checkpoint(runner, eval_res)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs
        )
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        return eval_res

    def checkpoint(self, runner, results):
        for metric in self.checkpoint_metrics:
            value = results[metric]
            old_value, old_epoch = self.best_model_metrics[metric]

            if value > old_value:
                if not self.out_dir:
                    self.out_dir = runner.work_dir

                # Removes old checkpoint
                ckpt_path = os.path.join(
                    self.out_dir,
                    self.filename_tmpl.format(metric, old_value).format(old_epoch),
                )
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)

                # Save new checkpoint
                template = self.filename_tmpl.format(metric, value)
                runner.save_checkpoint(
                    self.out_dir,
                    filename_tmpl=template,
                    save_optimizer=self.save_optimizer,
                )
                self.best_model_metrics[metric] = (value, runner.epoch + 1)


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self, dataloader, interval=1, gpu_collect=False, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                "dataloader must be a pytorch DataLoader, but got {}".format(
                    type(dataloader)
                )
            )
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import multi_gpu_test

        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, ".eval_hook"),
            gpu_collect=self.gpu_collect,
        )
        if runner.rank == 0:
            print("\n")
            self.evaluate(runner, results)
