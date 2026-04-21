import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from models.net_ewclora import Net
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency
from utils.covariance import OnlineCovariance


class ActMatD(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        self.topk = 1
        self.network = Net(args)

        self.gamma = args.get("gamma", 1.0)
        self.reg_weight = args["lambda"]
        self.omega_W = []
        self.count_updates = 0

    def after_task(self):
        super().after_task()

        print("=== Update Activation Covariance ===")
        self.count_updates += 1
        cov = CovarianceComputer(self.network, self.train_loader, self.device)
        cov_W = cov.compute_approx()

        omega_W_bk = self.omega_W[:]
        self.omega_W = []
        for idx in range(len(cov_W)):
            if len(omega_W_bk) != 0:
                self.omega_W.append(self.gamma * omega_W_bk[idx] + cov_W[idx])
            else:
                self.omega_W.append(cov_W[idx])

        self.network.accumulate_and_reset_lora()

    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)

        encoder_params = self.network.image_encoder.parameters()
        cls_params = [
            p
            for p in self.network.classifier_pool.parameters()
            if p.requires_grad == True
        ]

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

        encoder_params = {
            "params": encoder_params,
            "lr": self.lrate,
            "weight_decay": self.weight_decay,
        }
        cls_params = {
            "params": cls_params,
            "lr": self.fc_lrate,
            "weight_decay": self.weight_decay,
        }

        network_params = [encoder_params, cls_params]
        optimizer, scheduler = self.build_optimizer(network_params)
        check_params_consistency(self.network, optimizer)

        self._train_function(train_loader, optimizer, scheduler)

        if len(self.multiple_gpus) > 1:
            self.network = self.network.module
        return

    def _train_function(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes

                logits = self.network(inputs, use_new=True)["logits"]
                loss = F.cross_entropy(logits, targets)

                if self.count_updates != 0:
                    new_a_params = filter(
                        lambda p: getattr(p, "_is_new_a", False),
                        self.network.parameters(),
                    )
                    new_b_params = filter(
                        lambda p: getattr(p, "_is_new_b", False),
                        self.network.parameters(),
                    )
                    reg_loss = 0.0
                    for idx, (p_a, p_b) in enumerate(zip(new_a_params, new_b_params)):
                        delta_W = p_b @ p_a
                        C = self.omega_W[idx].type(torch.float32).to(self.device)
                        reg_loss += torch.sum((delta_W @ C) * delta_W)

                    loss += self.reg_weight / 2.0 * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self.cur_task,
                epoch + 1,
                self.epochs,
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            f"lora_new_A_k",
            f"lora_new_A_v",
            f"lora_new_B_k",
            f"lora_new_B_v",
        ]
        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))


class CovarianceComputer:
    def __init__(self, network, dataloader, device=torch.device("cpu")):
        self.model = network.to(device)
        self.dataloader = dataloader
        self.device = device

    def compute(self, max_batches=None):
        stores, handles = [], []

        def make_hook(store):
            def hook(mod, inp, out):
                x = inp[0]
                if x.dim() == 3:
                    B, T, D = x.shape
                    x_flat = x.reshape(B * T, D)
                else:
                    x_flat = x
                    D = x.shape[-1]
                if store["cobj"] is None:
                    store["cobj"] = OnlineCovariance(D, device=self.device, mode="sm")
                cobj = store["cobj"]
                x_flat = x_flat.to(self.device)
                cobj.C += x_flat.T @ x_flat
                cobj.n += x_flat.shape[0]

            return hook

        for module in self.model.modules():
            if hasattr(module, "qkv") and hasattr(module, "lora_new_A_k"):
                store = {"cobj": None}
                stores.append(store)
                handles.append(module.qkv.register_forward_hook(make_hook(store)))

        self.model.eval()
        with torch.no_grad():
            for i, (_, inputs, _) in enumerate(
                tqdm(self.dataloader, desc="Computing Covariances")
            ):
                if max_batches and i >= max_batches:
                    break
                inputs = inputs.to(self.device)
                self.model.forward(inputs, use_new=True)

        for h in handles:
            h.remove()

        cov_W = []
        for store in stores:
            C = store["cobj"].cov
            cov_W.append(C)  # k slot
            cov_W.append(C)  # v slot (same tensor, shared reference)
        return cov_W

    def compute_approx(self):
        """Data-free covariance proxy: C ≈ d^T d with d = B_new @ A_new.

        Walks modules in the same K-then-V order as compute(), producing a
        separate (Di, Di) proxy per K/V rather than the shared activation cov.
        """
        cov_W = []
        for module in self.model.modules():
            if hasattr(module, "lora_new_B_k") and hasattr(module, "lora_new_A_k"):
                d = (module.lora_new_B_k.weight @ module.lora_new_A_k.weight).detach()
                cov_W.append(d.T @ d)
            if hasattr(module, "lora_new_B_v") and hasattr(module, "lora_new_A_v"):
                d = (module.lora_new_B_v.weight @ module.lora_new_A_v.weight).detach()
                cov_W.append(d.T @ d)
        return cov_W
