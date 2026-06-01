import gc
import logging
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm

import optuna

from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer
from credit.preblock import ConcatPreblock, ERA5Normalizer, apply_preblocks
from credit.scheduler import update_on_batch
from credit.trainers.base_trainer import BaseTrainer
from credit.trainers.utils import accum_log, cycle

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int, conf: dict):
        """
        Trainer for ERA5 v2 data schema.

        Key differences from TrainerERA5 (v1):
          - Uses new nested data schema: conf["data"]["source"]["ERA5"]["variables"]
          - Applies a ConcatPreblock to assemble batch tensors before the model forward pass
            (no concat_and_reshape / reshape_only calls in the training loop)
          - forecast_len semantics: 1 = 1 step (v1 used 0 = 1 step)
          - backprop_on_timestep: range(1, forecast_len+1) instead of range(0, forecast_len+2)
          - Validation config read from conf["data_valid"] if present, else conf["data"]

        Args:
            model: The (possibly DDP/FSDP-wrapped) model.
            rank: Global rank of this process.
            conf: Full configuration dict.
        """
        super().__init__(model, rank, conf)
        logger.info("Loading ERA5-v2 trainer (new nested data schema, preblock-assembled batches)")

        # ---- Preblock: normalize then assemble batch field tensors into x and y ----
        preblocks = {}
        if conf.get("data", {}).get("scaler_type") == "std_new":
            preblocks["norm"] = ERA5Normalizer(conf)
        preblocks["concat"] = ConcatPreblock()
        self.preblocks = nn.ModuleDict(preblocks)

        # ---- Postblock conservation fixers ----
        post_conf = conf.get("model", {}).get("post_conf", {})
        self.water_conservation_weight = float(
            post_conf.get("global_water_fixer", {}).get("conservation_loss_weight", 0.0)
        )
        self.flag_mass_conserve = False
        self.flag_water_conserve = False
        self.flag_energy_conserve = False
        self.opt_mass = None
        self.opt_water = None
        self.opt_energy = None

        if post_conf.get("activate", False):
            if post_conf.get("global_mass_fixer", {}).get("activate", False) and post_conf["global_mass_fixer"].get(
                "activate_outside_model", False
            ):
                logger.info("Activate GlobalMassFixer outside of model")
                self.flag_mass_conserve = True
                self.opt_mass = GlobalMassFixer(post_conf)

            if post_conf.get("global_water_fixer", {}).get("activate", False) and post_conf["global_water_fixer"].get(
                "activate_outside_model", False
            ):
                logger.info("Activate GlobalWaterFixer outside of model")
                self.flag_water_conserve = True
                self.opt_water = GlobalWaterFixer(post_conf)

            if post_conf.get("global_energy_fixer", {}).get("activate", False) and post_conf["global_energy_fixer"].get(
                "activate_outside_model", False
            ):
                logger.info("Activate GlobalEnergyFixer outside of model")
                self.flag_energy_conserve = True
                self.opt_energy = GlobalEnergyFixer(post_conf)

        # ---- Data schema extraction (new nested schema) ----
        data_conf = conf["data"]
        source = next(iter(data_conf["source"].values()))
        vars_conf = source["variables"]
        prog = vars_conf.get("prognostic") or {}
        diag = vars_conf.get("diagnostic") or {}
        dyn = vars_conf.get("dynamic_forcing") or {}
        static_v = vars_conf.get("static") or {}
        num_levels = len(source.get("levels", []))

        # Diagnostic output channel count (excluded from autoregressive x update)
        # ERA5Dataset already flattens 3D: varnum_diag = vars_3D*levels + vars_2D
        self.varnum_diag = (len(diag.get("vars_3D", [])) * num_levels + len(diag.get("vars_2D", []))) if diag else 0

        # Forcing+static input channel count (last channels of x, not predicted by model)
        self.static_dim_size = len(dyn.get("vars_2D", [])) + len(static_v.get("vars_2D", []))

        self.retain_graph = data_conf.get("retain_graph", False)

        # forecast_len: 1 = 1 step (new semantics, unlike v1 where 0 = 1 step)
        self.forecast_len = data_conf["forecast_len"]
        if "backprop_on_timestep" in data_conf and data_conf["backprop_on_timestep"] is not None:
            self.backprop_on_timestep = data_conf["backprop_on_timestep"]
        elif data_conf.get("one_shot", False):
            # one_shot: only supervise the final rollout step. Intermediate steps are
            # run without loss so the model learns to produce stable multi-step rollouts
            # rather than fitting contradictory targets from unrelated batches.
            self.backprop_on_timestep = [self.forecast_len]
        else:
            self.backprop_on_timestep = list(range(1, self.forecast_len + 1))

        data_clamp = data_conf.get("data_clamp")
        if data_clamp is None:
            self.flag_clamp = False
            self.clamp_min = None
            self.clamp_max = None
        else:
            self.flag_clamp = True
            self.clamp_min = float(data_clamp[0])
            self.clamp_max = float(data_clamp[1])

        # Validation config: use data_valid block if present, else fall back to data
        data_valid = conf.get("data_valid", data_conf)
        self.valid_history_len = data_valid.get("history_len", data_conf.get("history_len", 1))
        self.valid_forecast_len = data_valid.get("forecast_len", self.forecast_len)

        # torch.compile — fuses ops across the forward pass for significant speedup
        if conf.get("trainer", {}).get("use_compile", False):
            logger.info("Compiling model with torch.compile() — first forward pass will be slow (~60s)")
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def train_one_epoch(self, epoch, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        """
        Train for one epoch.

        The inner loop iterates over forecast_len autoregressive steps. For each step:
          1. Pull the next batch from the dataloader (contains per-field tensors).
          2. Apply preblocks to assemble batch["x"] and batch["y"].
          3. For t > 1, replace the prognostic channels of x with the previous y_pred.
          4. Forward pass, optional postblock, loss and backprop on backprop_on_timestep.

        Args:
            epoch: Current epoch number.
            conf: Full configuration dict.
            trainloader: DataLoader for training.
            optimizer, criterion, scaler, scheduler, metrics: Standard training objects.

        Returns:
            dict: Training metrics for the epoch.
        """
        if self.ensemble_size > 1:
            logger.info(f"ensemble training with ensemble_size {self.ensemble_size}")
        logger.info(f"Using grad-max-norm value: {self.grad_max_norm}")

        # lambda scheduler steps once per epoch before batches
        if self.use_scheduler and self.scheduler_type == "lambda":
            scheduler.step()

        # resolve effective batches_per_epoch
        from torch.utils.data import IterableDataset

        batches_per_epoch = self.batches_per_epoch
        if not isinstance(trainloader.dataset, IterableDataset):
            if hasattr(trainloader.dataset, "batches_per_epoch"):
                dataset_batches = trainloader.dataset.batches_per_epoch()
            elif hasattr(trainloader.sampler, "batches_per_epoch"):
                dataset_batches = trainloader.sampler.batches_per_epoch()
            else:
                dataset_batches = len(trainloader)
            batches_per_epoch = (
                self.batches_per_epoch if 0 < self.batches_per_epoch < dataset_batches else dataset_batches
            )

        batch_group_generator = tqdm.tqdm(range(batches_per_epoch), total=batches_per_epoch, leave=True)
        self.model.train()

        if epoch == 0 and self.rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info("Model parameters: %.2fM total, %.2fM trainable", total_params / 1e6, trainable_params / 1e6)

        dl = cycle(trainloader)
        results_dict = defaultdict(list)
        _t_data_total = 0.0
        _t_compute_total = 0.0
        _t_step_start = time.perf_counter()

        for steps in range(batches_per_epoch):
            logs = {}
            loss = 0
            x_init = None  # snapshot of x at step 1 for GlobalMassFixer
            y_pred = None
            y = None

            for t in range(1, self.forecast_len + 1):
                _t0_data = time.perf_counter()
                batch = next(dl)
                _t_data_total += time.perf_counter() - _t0_data
                batch = apply_preblocks(self.preblocks, batch)

                if t == 1:
                    x = batch["x"].to(self.device).float()
                    if self.ensemble_size > 1:
                        x = torch.repeat_interleave(x, self.ensemble_size, 0)
                else:
                    # Roll x forward: take new batch's forcing/static, replace prog with y_pred
                    x_new = batch["x"].to(self.device).float()
                    if self.ensemble_size > 1:
                        x_new = torch.repeat_interleave(x_new, self.ensemble_size, 0)
                    n_prog = x_new.shape[1] - self.static_dim_size
                    y_pred_prog = y_pred[:, :n_prog, ...]
                    if not self.retain_graph:
                        y_pred_prog = y_pred_prog.detach()
                    x_new[:, :n_prog, ...] = y_pred_prog
                    x = x_new

                if self.flag_clamp:
                    x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

                _t0_compute = time.perf_counter()
                with torch.autocast(device_type="cuda", enabled=self.amp):
                    y_pred = self.model(x)

                if steps == 0 and t == 1:
                    logger.info(
                        "[rank%d] NaN/inf check — x nan=%s inf=%s | y_pred nan=%s inf=%s | y_pred abs max=%.3e",
                        self.rank,
                        torch.isnan(x).any().item(), torch.isinf(x).any().item(),
                        torch.isnan(y_pred).any().item(), torch.isinf(y_pred).any().item(),
                        y_pred.abs().max().item(),
                    )

                # postblock opts outside of model
                if self.flag_mass_conserve:
                    if t == 1:
                        x_init = x.clone()
                    input_dict = {"y_pred": y_pred, "x": x_init}
                    input_dict = self.opt_mass(input_dict)
                    y_pred = input_dict["y_pred"]

                water_conservation_loss = None
                if self.flag_water_conserve:
                    input_dict = {"y_pred": y_pred, "x": x}
                    input_dict = self.opt_water(input_dict)
                    y_pred = input_dict["y_pred"]
                    water_conservation_loss = input_dict.get("water_conservation_loss", None)
                    if water_conservation_loss is not None:
                        drift_pct = 100.0 * (water_conservation_loss.detach().float().sqrt().item())
                        accum_log(logs, {"water_drift_pct": drift_pct})

                if self.flag_energy_conserve:
                    input_dict = {"y_pred": y_pred, "x": x}
                    input_dict = self.opt_energy(input_dict)
                    y_pred = input_dict["y_pred"]

                # backprop on specified timesteps
                if t in self.backprop_on_timestep:
                    y = batch["y"].to(self.device).float()
                    if self.flag_clamp:
                        y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)

                    if steps == 0 and t == 1:
                        logger.info(
                            "[rank%d] Shapes — x: %s, y_pred: %s, y: %s",
                            self.rank, tuple(x.shape), tuple(y_pred.shape), tuple(y.shape),
                        )
                        logger.info(
                            "[rank%d] NaN check — y: %s, x: %s, y_pred: %s",
                            self.rank,
                            torch.isnan(y).any().item(),
                            torch.isnan(x).any().item(),
                            torch.isnan(y_pred).any().item(),
                        )
                    with torch.autocast(device_type="cuda", enabled=self.amp):
                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()
                        if water_conservation_loss is not None and self.water_conservation_weight > 0:
                            loss = loss + self.water_conservation_weight * water_conservation_loss
                    if steps == 0 and t == 1:
                        logger.info("[rank%d] loss value: %s", self.rank, loss.item())
                    accum_log(logs, {"loss": loss.item()})
                    scaler.scale(loss).backward(retain_graph=self.retain_graph)
                elif water_conservation_loss is not None and self.water_conservation_weight > 0:
                    # Intermediate steps (not in backprop_on_timestep): still penalise water-budget
                    # violations so the model cannot freely violate conservation at t=1 while
                    # optimising for t=2.  We do NOT add a supervised loss here — only the
                    # physics constraint.
                    with torch.autocast(device_type="cuda", enabled=self.amp):
                        conservation_only_loss = self.water_conservation_weight * water_conservation_loss
                    scaler.scale(conservation_only_loss).backward(retain_graph=True)

                _t_compute_total += time.perf_counter() - _t0_compute

                if self.distributed:
                    torch.distributed.barrier()

            # optimizer step
            scaler.unscale_(optimizer)
            if self.grad_max_norm == "dynamic":
                local_norm = torch.norm(
                    torch.stack([p.grad.detach().norm(2) for p in self.model.parameters() if p.grad is not None])
                )
                if self.distributed:
                    dist.all_reduce(local_norm, op=dist.ReduceOp.SUM)
                global_norm = local_norm.sqrt()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=global_norm)
            elif self.grad_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if self.ema is not None:
                self.ema.update(self.model)

            # log data-vs-compute breakdown every 50 steps
            if steps > 0 and steps % 50 == 0 and self.rank == 0:
                elapsed = time.perf_counter() - _t_step_start
                logger.info(
                    "[timing] steps=%d  data=%.1fs (%.0f%%)  compute=%.1fs (%.0f%%)  total=%.1fs  %.2f s/step",
                    steps,
                    _t_data_total, 100 * _t_data_total / elapsed,
                    _t_compute_total, 100 * _t_compute_total / elapsed,
                    elapsed, elapsed / steps,
                )

            # collect metrics every 10 steps (588 GPU→CPU syncs per call is expensive)
            # metrics every 10 steps — no all_reduce (per-rank is fine for logging,
            # and 588 sequential all_reduces was deadlocking DDP when ranks diverged)
            if steps % 10 == 0 and y_pred is not None and y is not None and self.rank == 0:
                metrics_dict = metrics(y_pred.detach().float(), y.detach().float())
                for name, value in metrics_dict.items():
                    # must call .item() — raw values are GPU tensors that accumulate
                    v = value.item() if hasattr(value, "item") else float(value)
                    results_dict[f"train_{name}"].append(v)

            batch_loss = torch.Tensor([logs.get("loss", 0.0)]).to(self.device)
            if self.distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())
            results_dict["train_forecast_len"].append(self.forecast_len)
            if self.flag_water_conserve and "water_drift_pct" in logs:
                results_dict.setdefault("train_water_drift_pct", []).append(logs["water_drift_pct"])

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                print(results_dict["train_loss"])
                raise optuna.TrialPruned()

            water_drift_str = ""
            if self.flag_water_conserve and results_dict.get("train_water_drift_pct"):
                water_drift_str = " water_drift: {:.2f}%".format(np.mean(results_dict["train_water_drift_pct"]))

            to_print = "Epoch: {} train_loss: {:.6f} train_acc: {:.6f} train_mae: {:.6f} forecast_len: {:.0f}{}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_acc"]),
                np.mean(results_dict["train_mae"]),
                self.forecast_len,
                water_drift_str,
            )
            if self.ensemble_size > 1:
                to_print += f" std: {np.mean(results_dict['train_std']):.6f}"
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.update(1)
                batch_group_generator.set_description(to_print)

            if self.use_scheduler and self.scheduler_type in update_on_batch:
                scheduler.step()

        batch_group_generator.close()
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(self, epoch, valid_loader, criterion, metrics):
        """
        Validate for one epoch.

        Runs self.valid_forecast_len autoregressive steps per sample.
        Loss and metrics are computed only at the final step (t == self.valid_forecast_len).

        Args:
            epoch: Current epoch number.
            conf: Full configuration dict.
            valid_loader: DataLoader for validation.
            criterion, metrics: Loss and metric callables.

        Returns:
            dict: Validation metrics for the epoch.
        """
        self.model.eval()

        from torch.utils.data import IterableDataset

        valid_batches_per_epoch = self.valid_batches_per_epoch
        if not isinstance(valid_loader.dataset, IterableDataset):
            if hasattr(valid_loader.dataset, "batches_per_epoch"):
                dataset_batches = valid_loader.dataset.batches_per_epoch()
            elif hasattr(valid_loader.sampler, "batches_per_epoch"):
                dataset_batches = valid_loader.sampler.batches_per_epoch()
            else:
                dataset_batches = len(valid_loader)
            valid_batches_per_epoch = (
                self.valid_batches_per_epoch if 0 < self.valid_batches_per_epoch < dataset_batches else dataset_batches
            )

        results_dict = defaultdict(list)
        batch_group_generator = tqdm.tqdm(range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True)

        dl = cycle(valid_loader)
        with torch.no_grad():
            for steps in range(valid_batches_per_epoch):
                y_pred = None
                y = None
                loss = 0
                x_init = None

                for t in range(1, self.valid_forecast_len + 1):
                    batch = next(dl)
                    batch = apply_preblocks(self.preblocks, batch)

                    if t == 1:
                        x = batch["x"].to(self.device).float()
                        if self.ensemble_size > 1:
                            x = torch.repeat_interleave(x, self.ensemble_size, 0)
                    else:
                        # Roll x forward for multi-step validation rollout
                        x_new = batch["x"].to(self.device).float()
                        if self.ensemble_size > 1:
                            x_new = torch.repeat_interleave(x_new, self.ensemble_size, 0)
                        n_prog = x_new.shape[1] - self.static_dim_size
                        # Trim to min batch size — consecutive next(dl) calls can yield different
                        # sizes when the dataset doesn't divide evenly (last mini-batch is smaller)
                        bs = min(x_new.shape[0], y_pred.shape[0])
                        x_new = x_new[:bs]
                        x_new[:, :n_prog, ...] = y_pred[:bs, :n_prog, ...].detach()
                        y_pred = y_pred[:bs]
                        x = x_new

                    if self.flag_clamp:
                        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

                    y_pred = self.model(x.float())

                    # postblock opts outside of model
                    if self.flag_mass_conserve:
                        if t == 1:
                            x_init = x.clone()
                        input_dict = {"y_pred": y_pred, "x": x_init}
                        input_dict = self.opt_mass(input_dict)
                        y_pred = input_dict["y_pred"]

                    if self.flag_water_conserve:
                        input_dict = {"y_pred": y_pred, "x": x}
                        input_dict = self.opt_water(input_dict)
                        y_pred = input_dict["y_pred"]

                    if self.flag_energy_conserve:
                        input_dict = {"y_pred": y_pred, "x": x}
                        input_dict = self.opt_energy(input_dict)
                        y_pred = input_dict["y_pred"]

                    # compute loss and metrics only at the final rollout step
                    if t == self.valid_forecast_len:
                        y = batch["y"].to(self.device).float()
                        if self.flag_clamp:
                            y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)
                        # Align y with y_pred in case of a partial last batch
                        if y.shape[0] != y_pred.shape[0]:
                            y = y[: y_pred.shape[0]]

                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()
                        metrics_dict = metrics(y_pred.float(), y.float())
                        for name, value in metrics_dict.items():
                            value = torch.Tensor([value]).to(self.device, non_blocking=True)
                            if self.distributed:
                                dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                            results_dict[f"valid_{name}"].append(value[0].item())

                batch_loss = torch.Tensor([loss.item() if torch.is_tensor(loss) else loss]).to(self.device)
                if self.distributed:
                    torch.distributed.barrier()

                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(self.valid_forecast_len)

                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"]),
                )
                if self.ensemble_size > 1:
                    to_print += f" std: {np.mean(results_dict['valid_std']):.6f}"
                if self.rank == 0:
                    batch_group_generator.update(1)
                    batch_group_generator.set_description(to_print)

        batch_group_generator.close()

        if self.distributed:
            torch.distributed.barrier()

        torch.cuda.empty_cache()
        gc.collect()

        return results_dict
