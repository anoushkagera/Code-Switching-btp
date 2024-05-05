from torch import optim


class AdamInverseSqrtWithWarmup(optim.Adam):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup:

      lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
      lr = lrs[update_num]

    After warmup:

      lr = decay_factor / sqrt(update_num)

    where

      decay_factor = lr * sqrt(warmup_updates)
    """
    
    # This line defines the initialization method for the optimizer. 
    # It accepts various parameters including params (iterable of parameters to optimize),
    # lr (learning rate), betas (coefficients used for computing running averages of gradient 
    # and its square), eps (term added to the denominator to improve numerical stability), 
    # weight_decay (weight decay coefficient for regularization), warmup_updates (number of warm-up updates), 
    # and warmup_init_lr (initial learning rate during warm-up

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr

        # linearly warmup for the first warmup_updates
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates**0.5

        self._num_updates = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates*self.lr_step
        else:
            return self.decay_factor * num_updates**-0.5

    def step(self, closure=None):
        super().step(closure)
        self._num_updates += 1

        # update learning rate
        new_lr = self.get_lr_for_step(self._num_updates)
        for param_group in self.param_groups:
            param_group['lr'] = new_lr

# Adaptive Learning Rates: 
# Adam dynamically adjusts the learning rates 
# for each parameter based on the past gradients and their square averages. 
# This helps in converging faster and more reliably, especially in the presence of sparse gradients.
# Bias Correction:
#   Adam performs bias correction to counteract the tendency of the moving averages to be biased towards zero at the beginning of training.
# Efficient Optimization:
#   Adam is computationally efficient and has been widely used in various deep learning applications.