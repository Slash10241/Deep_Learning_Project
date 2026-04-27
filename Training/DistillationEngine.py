import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from TrainingEngine import TrainingEngine

class DistillationEngine(TrainingEngine):
    """
    Knowledge Distillation execution pipeline.

    Inherits from TrainingEngine, while overriding the training pass to implement a blended hard/soft target loss.
    """

    def __init__(
        self,
        expert_network: nn.Module,#The frozen pre-trained teacher model
        *args,
        kd_temp: float = 3.0,#The temperature 'T' used to soften logits. Default: 3.0.
        hard_label_weight: float = 0.5,#The alpha factor weighting the standard CE loss. Default: 0.5.
        **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.kd_temp = kd_temp
        self.hard_label_weight = hard_label_weight

        self.expert_network = expert_network.to(self.compute_device)
        self.expert_network.eval()
        
        self.expert_network.requires_grad_(False)

        self.divergence_metric = nn.KLDivLoss(reduction='batchmean')


    def train_one_epoch(
        self,
        epoch_index: int,
        log_interval: Optional[int] = 50,
    ) -> Dict[str, float]:
        """
        Executes distillation training pass.
        """
        self.network.train()
        self._clear_trackers()

        num_steps = len(self.data_iterator)

        hw_backend = 'cuda' if self.compute_device.type == 'cuda' else 'cpu'
        is_cuda_active = (hw_backend == 'cuda')
        
        amp_ctx = torch.autocast(device_type=hw_backend, enabled=is_cuda_active)

        for step_num, (inputs, targets) in enumerate(self.data_iterator):
            
            inputs = inputs.to(
                device=self.compute_device, 
                memory_format=torch.channels_last, 
                non_blocking=True
            )
            targets = targets.to(self.compute_device, non_blocking=True)

            self.opt.zero_grad(set_to_none=True)

            with amp_ctx:
                with torch.no_grad():
                    expert_preds = self.expert_network(inputs)

                student_preds = self.network(inputs)

                hard_loss = self.loss_fn(student_preds, targets)

                scaled_student_logs = F.log_softmax(student_preds/self.kd_temp, dim=-1)
                scaled_expert_probs = F.softmax(expert_preds/self.kd_temp, dim=-1)
                
                soft_loss = self.divergence_metric(scaled_student_logs, scaled_expert_probs) * (self.kd_temp ** 2)
                combined_loss = (self.hard_label_weight * hard_loss) + ((1.0 - self.hard_label_weight) * soft_loss)

            self.grad_scaler.scale(combined_loss).backward()
            
            pre_step_scale = self.grad_scaler.get_scale()
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            
            post_step_scale = self.grad_scaler.get_scale()

            if self.scheduler is not None:
                if pre_step_scale <= post_step_scale:
                    self.scheduler.step()
                else:
                    print("GradScaler detected overflow")

            self._accumulate_stats(combined_loss, student_preds, targets)

            if log_interval and (step_num + 1) % log_interval == 0:
                current_metrics = self._calculate_epoch_stats()
                print(
                    f"[Epoch {epoch_index:03d}] "
                    f"Step {step_num + 1:04d}/{num_steps:04d} | "
                    f"Loss: {current_metrics['epoch_loss']:.4f} | "
                    f"Acc: {current_metrics['epoch_acc']:.4f} | "
                    f"CE: {hard_loss.item():.4f} | KD: {soft_loss.item():.4f}"
                )

        epoch_summary = self._calculate_epoch_stats()
        print(
            f"[Epoch {epoch_index:03d}] finished | "
            f"Avg Loss: {epoch_summary['epoch_loss']:.4f} | "
            f"Avg Acc: {epoch_summary['epoch_acc']:.4f}"
        )
        
        return epoch_summary