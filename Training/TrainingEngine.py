import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

class TrainingEngine:
    """
    A modular execution engine for image classification. 
    It handles the core training loop (with speed optimizations). 
    It is designed to train both ViTs and ResNets. 
    """

    def __init__(
        self,
        network: nn.Module, #The neural network model (ViT or ResNet).
        data_iterator: DataLoader,#PyTorch DataLoader providing training batches.
        loss_fn: nn.Module,#The objective function used to calculate loss.
        opt: torch.optim.Optimizer,#The optimizer updating the network weights.
        compute_device: torch.device,#The hardware device (cuda or cpu).
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,#(Optional) Learning rate scheduler for step decay.
        batch_aug: Optional[object] = None,#for cutmix / mixup
    ) -> None:
        
        self.network = network.to(compute_device)
        self.data_iterator = data_iterator
        self.loss_fn = loss_fn
        self.opt = opt
        self.compute_device = compute_device
        self.scheduler = scheduler
        self.batch_aug = batch_aug
        
        amp_device_str = 'cuda' if self.compute_device.type == 'cuda' else 'cpu'
        self.grad_scaler = torch.amp.GradScaler(
            device=amp_device_str, 
            enabled=self.compute_device.type == 'cuda'
        )

        self.total_loss: float = 0.0
        self.correct_preds: int = 0
        self.total_samples: int = 0

    def _clear_trackers(self) -> None:
        """Resets epoch-level accumulators."""
        self.total_loss = 0.0
        self.correct_preds = 0
        self.total_samples = 0

    @torch.inference_mode() 
    def _accumulate_stats(
        self,
        batch_loss: torch.Tensor,
        preds_raw: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> None:
        """Aggregates batch statistics into epoch totals."""
        b_size = y_batch.size(0)
        self.total_loss += batch_loss.item() * b_size

        class_preds = preds_raw.argmax(dim=-1)
        
        if y_batch.ndim > 1:
            y_true = y_batch.argmax(dim=-1)
        else:
            y_true = y_batch
            
        self.correct_preds += torch.eq(class_preds, y_true).sum().item()
        self.total_samples += b_size

    def _calculate_epoch_stats(self) -> Dict[str, float]:
        """Computes and returns the final averaged metrics for the epoch."""
        denominator = max(self.total_samples, 1)
        return {
            "epoch_loss": self.total_loss / denominator,
            "epoch_acc": self.correct_preds / denominator
        }


    def train_one_epoch(
        self,
        epoch_num: int,
        print_freq: Optional[int] = 50,
    ) -> Dict[str, float]:
        """
        Executes a single training pass over data_iterator.
        """
        self.network.train()
        self._clear_trackers()
 
        amp_device_type = 'cuda' if self.compute_device.type == 'cuda' else 'cpu'
 
        pbar = tqdm(
            self.data_iterator,
            desc=f"[Epoch {epoch_num:03d}] Train",
            unit="batch",
            dynamic_ncols=True,
            colour="green",
        )
 
        for step_idx, (x_batch, y_batch) in enumerate(pbar):
            x_batch = x_batch.to(self.compute_device, non_blocking=True)
            y_batch = y_batch.to(self.compute_device, non_blocking=True)

            if self.batch_aug is not None:
                x_batch, y_batch = self.batch_aug(x_batch, y_batch)
 
            self.opt.zero_grad(set_to_none=True)
 
            with torch.autocast(device_type=amp_device_type, enabled=self.compute_device.type == 'cuda'):
                preds_raw = self.network(x_batch)
                batch_loss = self.loss_fn(preds_raw, y_batch)
 
            self.grad_scaler.scale(batch_loss).backward()
            scale_before = self.grad_scaler.get_scale()
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            scale_after = self.grad_scaler.get_scale()
 
            if self.scheduler is not None:
                if scale_before <= scale_after:
                    self.scheduler.step()
                else:
                    tqdm.write("Warning: Gradient overflow — scheduler step skipped.")
 
            self._accumulate_stats(batch_loss, preds_raw, y_batch)
 
            # update progress bar with running stats
            current_stats = self._calculate_epoch_stats()
            pbar.set_postfix(
                loss=f"{current_stats['epoch_loss']:.4f}",
                acc=f"{current_stats['epoch_acc']*100:.2f}%",
            )
 
            if print_freq and (step_idx + 1) % print_freq == 0:
                tqdm.write(
                    f"  [Epoch {epoch_num:03d}] "
                    f"Step {step_idx + 1:04d}/{len(pbar):04d} | "
                    f"loss={current_stats['epoch_loss']:.4f} | "
                    f"acc={current_stats['epoch_acc']:.4f}"
                )
 
        final_stats = self._calculate_epoch_stats()
        tqdm.write(
            f"[Epoch {epoch_num:03d}] COMPLETED | "
            f"avg_loss={final_stats['epoch_loss']:.4f} | "
            f"avg_acc={final_stats['epoch_acc']:.4f}"
        )
        return final_stats
 
    @torch.inference_mode()
    def evaluate(
        self,
        inference_loader: DataLoader,
        phase_label: str = "Validation",
    ) -> Dict[str, float]:
        """
        Runs a gradient-free evaluation pass over the dataset.
 
        Computes running loss and accuracy, plus Macro Precision, Recall, F1
        computed once at the end of the epoch over the full set of predictions.
        """
        from sklearn.metrics import precision_recall_fscore_support
 
        self.network.eval()
        self._clear_trackers()
 
        hardware_backend = 'cuda' if self.compute_device.type == 'cuda' else 'cpu'
        is_cuda_active   = (hardware_backend == 'cuda')
        amp_context      = torch.autocast(device_type=hardware_backend, enabled=is_cuda_active)
 
        all_preds:   list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
 
        pbar = tqdm(
            inference_loader,
            desc=f"[{phase_label.capitalize()}]      ",
            unit="batch",
            dynamic_ncols=True,
            colour="blue",
        )
 
        for features, targets in pbar:
            features = features.to(
                device=self.compute_device,
                memory_format=torch.channels_last,
                non_blocking=True,
            )
            targets = targets.to(device=self.compute_device, non_blocking=True)
 
            with amp_context:
                logits    = self.network(features)
                step_cost = self.loss_fn(logits, targets)
 
            self._accumulate_stats(step_cost, logits, targets)
 
            batch_preds = logits.argmax(dim=-1).detach().cpu()
            batch_targets = (
                targets.argmax(dim=-1).detach().cpu()
                if targets.ndim > 1
                else targets.detach().cpu()
            )
            all_preds.append(batch_preds)
            all_targets.append(batch_targets)
 
            current_stats = self._calculate_epoch_stats()
            pbar.set_postfix(
                loss=f"{current_stats['epoch_loss']:.4f}",
                acc=f"{current_stats['epoch_acc']*100:.2f}%",
            )
 
        phase_results = self._calculate_epoch_stats()
 
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
 
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average='macro',
            zero_division=0,
        )
 
        phase_results["macro_precision"] = float(macro_precision)
        phase_results["macro_recall"]    = float(macro_recall)
        phase_results["macro_f1"]        = float(macro_f1)
 
        tqdm.write(
            f"[{phase_label.upper()}] metrics | "
            f"Mean Loss: {phase_results['epoch_loss']:.4f} | "
            f"Accuracy: {phase_results['epoch_acc'] * 100:.2f}% | "
            f"Macro F1: {phase_results['macro_f1'] * 100:.2f}% | "
            f"Macro P:  {phase_results['macro_precision'] * 100:.2f}% | "
            f"Macro R:  {phase_results['macro_recall'] * 100:.2f}%"
        )
 
        return phase_results