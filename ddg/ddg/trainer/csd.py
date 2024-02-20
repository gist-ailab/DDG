import time
import torch
import numpy as np
from ddg.trainer import TrainerDG


__all__ = ['DomainMix']


class CSD(TrainerDG):
    """CSD trainer provided by (D)DG.
    """

    def __init__(self):
        super(CSD, self).__init__()
        self.mix_type = self.args.domain_mix_type
        self.alpha = self.args.domain_mix_alpha
        self.beta = self.args.domain_mix_beta
        self.dist_beta = torch.distributions.Beta(self.alpha, self.beta)
    
    def add_extra_args(self):
        super(self).add_extra_args()
        parse = self.parser
        parse.add_argument('--domain', type=str, default='crossdomain', choices={'random', 'crossdomain'},
                        help='Mix type for DomainMix.')
    # def add_extra_args(self):
    #     super(CSD, self).add_extra_args()
    #     parse = self.parser
    #     parse.add_argument('--domain-mix-type', type=str, default='crossdomain', choices={'random', 'crossdomain'},
    #                        help='Mix type for DomainMix.')
    #     parse.add_argument('--domain-mix-alpha', type=float, default=1.0, help='alpha for DomainMix.')
    #     parse.add_argument('--domain-mix-beta', type=float, default=1.0, help='beta for DomainMix.')
    def run_epoch(self):
        args = self.args
        progress = self.progress['train']
        end = time.time()
        for i, ((data, class_1), d_idx) in enumerate(self.data_loaders['train']):
            # measure data loading time

            oh_dids = torch.tensor(one_hot(d_idx, NUM_DOMAINS), dtype=torch.float, device='cuda')
            data_time = time.time() - end

            loss, acc1, acc5, batch_size = self.model_forward_backward(data, on_dids)

            batch_time = time.time() - end
            self.meters_update(batch_time=batch_time,
                               data_time=data_time,
                               losses=loss.item(),
                               top1=acc1[0],
                               top5=acc5[0],
                               batch_size=batch_size)

            # measure elapsed time
            end = time.time()

            if i % args.log_freq == 0:
                progress.display(i)

    def model_forward_backward(self, batch):
        images, target, label_a, label_b, lam = self.parse_batch_train(batch)
        output = self.model_inference(images)
        loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)
        self.optimizer_step(loss)
        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
        return loss, acc1, acc5, images.size(0)

    def parse_batch_train(self, batch):
        images, target, domain = super(CSD, self).parse_batch_train(batch)
        images, target_a, target_b, lam = self.domain_mix(images, target, domain)
        return images, target, target_a, target_b, lam

    def one_hot(ids, depth):
        z = np.zeros([len(ids), depth])
        z[np.arange(len(ids)), ids] = 1
        return z

    # def domain_mix(self, x, target, domain):
    #     lam = (self.dist_beta.rsample((1,)) if self.alpha > 0 else torch.tensor(1)).to(x.device)

    #     # random shuffle
    #     perm = torch.randperm(x.size(0), dtype=torch.int64, device=x.device)
    #     if self.mix_type == 'crossdomain':
    #         domain_list = torch.unique(domain)
    #         if len(domain_list) > 1:
    #             for idx in domain_list:
    #                 cnt_a = torch.sum(domain == idx)
    #                 idx_b = (domain != idx).nonzero().squeeze(-1)
    #                 cnt_b = idx_b.shape[0]
    #                 perm_b = torch.ones(cnt_b).multinomial(num_samples=cnt_a, replacement=bool(cnt_a > cnt_b))
    #                 perm[domain == idx] = idx_b[perm_b]
    #     elif self.mix_type != 'random':
    #         raise NotImplementedError(f"Chooses {'random', 'crossdomain'}, but got {self.mix_type}.")
    #     mixed_x = lam * x + (1 - lam) * x[perm, :]
    #     target_a, target_b = target, target[perm]
    #     return mixed_x, target_a, target_b, lam


if __name__ == '__main__':
    CSD().run()
