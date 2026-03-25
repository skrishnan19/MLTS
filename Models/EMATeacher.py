import torch
import torch.nn as nn
from copy import deepcopy
import math


class EMATeacher:
    """
    Teacher model that maintains an EMA of the student model weights
    - Updates parameters with EMA: teacher = decay * teacher + (1 - decay) * student
    - Updates buffers (like BatchNorm statistics) with direct copy from student
    - Can be used directly for evaluation: teacher(input)
    """

    def __init__(self, student_model, decay, tot_itr):
        # Create teacher as a deep copy of the student
        self.teacher = deepcopy(student_model)

        # Disable gradients for teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.decay = decay
        self.totalItr = tot_itr
        self.student_model = student_model

        # Initialize teacher weights to match student
        self.num_updates = 0
        self._update_teacher(init=True)
  
    def get_exp_momentum(self, base_momentum=0.9, current_step=0, lambda_rate=10):
        progress = current_step / self.totalItr
        momentum = self.decay - (self.decay - base_momentum) * math.exp(-lambda_rate * progress)
        return momentum

    def _update_teacher(self, init=False):
        """
        Update teacher weights using student weights
        - init: If True, copy directly without EMA (for initialization)
        """
        self.num_updates += 1
        m = self.get_exp_momentum(current_step=self.num_updates)
        with torch.no_grad():
            # Update parameters with EMA
            student_params = dict(self.student_model.named_parameters())
            teacher_params = dict(self.teacher.named_parameters())

            for name, param in teacher_params.items():
                if init:
                    param.data.copy_(student_params[name].data)
                else:
                    param.data.mul_(m).add_(
                        student_params[name].data,
                        alpha=1 - m
                    )

            # Update buffers (like BatchNorm stats) with direct copy
            student_buffers = dict(self.student_model.named_buffers())
            teacher_buffers = dict(self.teacher.named_buffers())

            for name, buffer in teacher_buffers.items():
                if name in student_buffers:  # Safety check
                    buffer.data.copy_(student_buffers[name].data)

    def update(self):
        """Update teacher weights using EMA of student weights"""
        self._update_teacher()

    def __call__(self, *args, **kwargs):
        """Enable direct calling like teacher(input)"""
        return self.teacher(*args, **kwargs)

    def eval(self):
        """Set teacher to evaluation mode"""
        self.teacher.eval()

    def to(self, device):
        """Move teacher to specified device"""
        self.teacher.to(device)
        return self

    @property
    def device(self):
        """Get device of teacher model"""
        return next(self.teacher.parameters()).device

    def get_num_updates(self):
        return self.num_updates