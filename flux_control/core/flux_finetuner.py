import math
import os
import shutil
import logging
import random
import pickle
from typing import Annotated, Literal, Any, cast

import aim
import torch
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.logging import RichHandler
from pydantic import BaseModel, PlainValidator, model_validator
from accelerate import Accelerator, DistributedType
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
import transformers
import diffusers
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_control import FluxControlPipeline
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.optimization import get_scheduler

from ..adapters import BaseAdapter, parse_adapter_config
from ..datasets import parse_dataset
from ..utils.common import flatten_dict
from ..utils.upcasting import (
    apply_layerwise_upcasting,
    cast_trainable_parameters,
    LayerwiseUpcastingGranularity,
)
from .sampler import FluxSampler


logger = get_logger(__name__)


class FluxFinetunerProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            if task.fields.get("progress_type") == "train":
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    TextColumn("• Epoch: {task.fields[epoch]}"),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    TextColumn("• Loss: {task.fields[loss]:.4f}"),
                    TextColumn("• LR: {task.fields[lr]:.6f}"),
                )
            elif task.fields.get("progress_type") == "sample":
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                )
            yield self.make_tasks_table([task])


class FluxFinetuner(BaseModel):
    adapter: Annotated[BaseAdapter, PlainValidator(parse_adapter_config)]
    sampler: FluxSampler
    dataset: dict[str, Any]
    dataloader_num_workers: int = 0

    # --- Loading and Saving ---
    output_dir: str
    logging_dir: str = "./runs"
    experiment_name: str | None = None
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    resume_from_checkpoint: str | None = None
    checkpointing_steps: int = 500
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    sample_steps: int = 50
    sample_batch_pickle: str | None = None
    checkpointings_limit: int | None = None
    _resume_checkpoint_path: str | None = None
    _resume_checkpoint_step: int = 0
    _sample_batch: dict[str, dict] | None = None
    """
    Path to the checkpoint to resume from, relative to the output_dir. Will try to read the
    training step from the file path.  If "latest", will try to find the latest checkpoint
    in the output_dir. If None, training will start from scratch.
    """

    @model_validator(mode="after")
    def _check_resume_from_checkpoint(self):
        if self.resume_from_checkpoint is None:
            return self

        if self.resume_from_checkpoint != "latest":
            path = os.path.basename(self.resume_from_checkpoint)
        else:
            dirs = os.listdir(self.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=self._parse_checkpoint_step)
            path = dirs[-1] if len(dirs) > 0 else None
            if path is None:
                raise ValueError(f"Checkpoint {path} not found")

        self._resume_checkpoint_step = self._parse_checkpoint_step(path)
        if self._resume_checkpoint_step < 0:
            raise ValueError(f"Cannot parse checkpoint step from {path}")
        self._resume_checkpoint_path = os.path.join(self.output_dir, path)

        return self

    @model_validator(mode="after")
    def _check_checkpointing_steps(self):
        if self.checkpointing_steps % self.gradient_accumulation_steps != 0:
            raise ValueError(
                f"Checkpointing steps {self.checkpointing_steps} must be divisible by gradient accumulation steps {self.gradient_accumulation_steps}"
            )
        return self

    @model_validator(mode="after")
    def _check_sample_batch(self):
        if self.sample_batch_pickle is None:
            return self
        if self.sample_steps % self.gradient_accumulation_steps != 0:
            raise ValueError(
                f"Sample steps {self.sample_steps} must be divisible by gradient accumulation steps {self.gradient_accumulation_steps}"
            )
        with open(self.sample_batch_pickle, "rb") as f:
            self._sample_batch = pickle.load(f)
        # Make sure the sample batch is a dict of dicts
        if not isinstance(self._sample_batch, dict):
            raise ValueError(
                f"Sample batch must be a dict, got {type(self._sample_batch)}"
            )
        for k, v in self._sample_batch.items():
            if not isinstance(v, dict):
                raise ValueError(f"Sample batch must be a dict of dicts, got {type(v)}")
        return self

    # --- Precision and Memory Optimization ---
    accelerator_amp_mode: Literal["no", "bf16"] = "no"
    """
    Accelerator mixed precision mode. Can be "no" or "bf16".
    
    1. "no": No mixed precision.
    2. "bf16": Use bf16 mixed precision, in which case base_precision must be "bf16" and
    trainable_precision must be "fp32".
    
    Note that "fp16" is not supported since Flux.1 Dev is native bf16.
    """
    base_precision: Literal["fp32", "bf16", "fp8-upcast"] = "fp8-upcast"
    """
    Precision for the base transformer model.
    
    1. "fp32": Use fp32 precision, cost ~48GB VRAM for weights. This is not very useful since 
    the model is already in bf16.
    2. "bf16": Use bf16 precision, cost ~23GB VRAM. Not feasible for 24GB GPUs.
    3. "fp8-upcast": Store weights in fp8 and compute in bf16, cost ~11GB VRAM. Also speeds up
    the training.
    
    Currently quantization (store & compute in fp8) is not supported.
    """
    trainable_precision: Literal["fp32", "bf16"] = "bf16"
    """
    Precision for the trainable parameters. can be "fp32" or "bf16".
    """
    allow_tf32: bool = False
    gradient_checkpointing: bool = True
    """
    Effectively reduces the memory usage for saving activations (use ~3GB for bf16). Double the
    forward pass time.
    """
    use_8bit_adam: bool = False
    """
    Store the momentum and variance of Adam in 8-bit. Requires bitsandbytes package.
    """
    _weight_dtype: torch.dtype = torch.bfloat16
    _trainable_dtype: torch.dtype = torch.bfloat16

    @model_validator(mode="after")
    def _check_precision(self):
        if self.accelerator_amp_mode == "bf16":
            if self.base_precision != "bf16" or self.trainable_precision != "fp32":
                raise ValueError(
                    "When using accelerator's bf16 amp mode, base_precision must be bf16 and trainable_precision must be fp32"
                )
        return self

    # --- Hyperparameters ---
    seed: int = random.randint(0, 2**32 - 1)
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 1
    train_steps: int | None = None
    """
    Number of total training steps. If None, train_epochs must be set.
    """
    train_epochs: int | None = None
    """
    Number of total training epochs. If None, train_steps must be set.
    """
    _train_steps: int = 0
    _train_epochs: int = 0

    @model_validator(mode="after")
    def _check_train_steps(self):
        if self.train_steps is None and self.train_epochs is None:
            raise ValueError("Either train_steps or train_epochs must be set")
        if self.train_steps is not None and self.train_epochs is not None:
            raise ValueError("Only one of train_steps or train_epochs can be set")
        return self

    learning_rate: float = 5e-6
    scale_lr: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8

    lr_scheduler: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    weighting_scheme: Literal[
        "sigma_sqrt", "logit_normal", "mode", "cosmap", "none"
    ] = "none"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    guidance_scale: float = 3.5

    max_grad_norm: float = 1.0

    _accelerator: Accelerator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._accelerator = self._make_accelerator()
        self._initialize_logging(kwargs)

        self._weight_dtype = (
            torch.float32 if self.base_precision == "fp32" else torch.bfloat16
        )
        self._trainable_dtype = (
            torch.float32 if self.trainable_precision == "fp32" else torch.bfloat16
        )

    def _make_accelerator(self) -> Accelerator:
        return Accelerator(
            mixed_precision="no",
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            log_with="aim",
            project_config=ProjectConfiguration(
                project_dir=self.output_dir,
                logging_dir=self.logging_dir,
            ),
        )

    def _initialize_logging(self, config: dict):
        rich_handler = RichHandler()
        transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.disable_progress_bar()
        transformers.utils.logging.add_handler(rich_handler)
        diffusers.utils.logging.disable_default_handler()
        diffusers.utils.logging.add_handler(rich_handler)
        diffusers.utils.logging.disable_progress_bar()
        logging.basicConfig(
            level=self.log_level,
            handlers=[rich_handler],
        )
        logger.info(self._accelerator.state, main_process_only=False)  # type: ignore
        if self._accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        experiment_name = (
            os.path.basename(self.output_dir)
            if self.experiment_name is None
            else self.experiment_name
        )
        self._accelerator.init_trackers(
            experiment_name,
            config=flatten_dict(config),
        )
        if self._accelerator.is_main_process:
            self._info(f"Saving logs to {self.logging_dir}")
            self._info(f"Saving model to {self.output_dir}")
            self._info(f"Full config: {config}")

    def _info(self, message: str):
        if self._accelerator.is_main_process:
            logger.info(message)

    def _debug(self, message: str):
        if self._accelerator.is_main_process:
            logger.debug(message)

    def _make_transformer(self) -> FluxTransformer2DModel:
        transformer = cast(
            FluxTransformer2DModel,
            FluxTransformer2DModel.from_pretrained(
                self.pretrained_model_id,
                subfolder="transformer",
                torch_dtype=self._weight_dtype,
            ),
        )
        transformer.requires_grad_(False)
        self.adapter.install_modules(transformer)
        cast_trainable_parameters(transformer, self._trainable_dtype)
        self._info(
            f"Transformer model created with {self.pretrained_model_id} and {self.adapter.__class__.__name__}"
        )
        return transformer

    def _unwrap_model(self, model):
        model = self._accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def _save_model_hook(self, models, weights, output_dir):
        if not self._accelerator.is_main_process:
            return

        assert len(models) == 1
        model = self._unwrap_model(models[0])
        assert isinstance(model, FluxTransformer2DModel)
        layers_to_save = self.adapter.save_model(model)

        if weights:
            # Remove the model from input list to avoid default saving behavior
            weights.pop()

        # We save the lora with FluxControlPipeline so it can be directly loaded
        # by the pipeline for inference
        FluxControlPipeline.save_lora_weights(
            output_dir, transformer_lora_layers=layers_to_save
        )

        self._info(f"Saved model to {output_dir}")

    def _load_model_hook(self, models, input_dir):
        if self._accelerator.distributed_type == DistributedType.DEEPSPEED:
            model = self._make_transformer()
        else:
            assert len(models) == 1
            assert isinstance(self._unwrap_model(models[0]), FluxTransformer2DModel)
            # Remove the model from input list to avoid default loading behavior
            model = models.pop()

        lora_state_dict = cast(dict, FluxControlPipeline.lora_state_dict(input_dir))
        self.adapter.load_model(model, lora_state_dict)
        cast_trainable_parameters(model, self._trainable_dtype)
        self._info(f"Loaded model from {input_dir}")

    def _optimize_model(self, transformer):
        if self.base_precision == "fp8-upcast":
            apply_layerwise_upcasting(
                transformer,
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=self._weight_dtype,
                granularity=LayerwiseUpcastingGranularity.PYTORCH_LAYER,
            )
            self._info(f"Applied layerwise upcasting.")

        if self.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
            self._info(f"Gradient checkpointing enabled.")

        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            self._info(f"TF32 enabled.")

    def _make_optimizer(self, transformer) -> torch.optim.Optimizer:
        learn_rate = (
            (
                self.learning_rate
                * self.gradient_accumulation_steps
                * self.train_batch_size
                * self._accelerator.num_processes
            )
            if self.scale_lr
            else self.learning_rate
        )

        if self.use_8bit_adam:
            try:
                from bitsandbytes.optim import Adam8bit
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes package."
                )
            optimizer_class = Adam8bit
        else:
            optimizer_class = torch.optim.AdamW

        parameters = list(
            filter(
                lambda p: p.requires_grad,
                transformer.parameters(),
            )
        )

        optimzer = optimizer_class(
            parameters,
            lr=learn_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,  # type: ignore
            eps=self.adam_epsilon,
        )

        num_trainable_params = sum(p.numel() for p in parameters)
        self._info(
            f"{optimzer.__class__.__name__} created with {num_trainable_params} trainable parameters"
        )
        return optimzer

    def _make_dataloader(self) -> torch.utils.data.DataLoader:
        with self._accelerator.main_process_first():
            dataset = parse_dataset(self.dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
        )
        self._info(f"DataLoader created with {len(dataset)} samples")  # type: ignore
        return dataloader

    def _calculate_real_train_steps(self, dataloader):
        step_per_epoch = math.ceil(len(dataloader) / self._accelerator.num_processes)
        if self.train_epochs is not None:
            self._train_steps = self.train_epochs * step_per_epoch
            self._train_epochs = cast(int, self.train_epochs)
        else:
            self._train_steps = cast(int, self.train_steps)
            self._train_epochs = math.ceil(self._train_steps / step_per_epoch)
        self._info(f"Training for {self._train_steps} steps ({self._train_epochs} epochs)")

    def _make_lr_scheduler(self, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        lr_scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps * self._accelerator.num_processes,
            num_training_steps=self._train_steps * self._accelerator.num_processes,
            num_cycles=self.lr_num_cycles,
            power=self.lr_power,
        )
        self._info(f"Created {lr_scheduler.__class__.__name__} scheduler")
        return lr_scheduler

    def _parse_checkpoint_step(self, checkpoint: str) -> int:
        try:
            return int(checkpoint.split("-")[-1])
        except:
            return -1

    def _try_resume_from_checkpoint(self) -> int:
        if self._resume_checkpoint_path is None:
            return 0
        self._accelerator.load_state(self._resume_checkpoint_path)
        self._info(f"Resumed from checkpoint {self._resume_checkpoint_path}")
        return self._resume_checkpoint_step

    def _train_step(self, transformer, batch) -> torch.Tensor:
        batch_size = batch["clean_latents"].shape[0]
        timesteps = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        ).to(device=batch["clean_latents"].device, dtype=self._weight_dtype)
        if self._unwrap_model(transformer).config.guidance_embeds:
            guidance = torch.full(
                (batch_size,),
                self.guidance_scale,
                device=batch["clean_latents"].device,
                dtype=self._weight_dtype,
            )
        else:
            guidance = None
        loss = self.adapter.train_step(transformer, batch, timesteps, guidance)
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme, sigmas=timesteps
        ).to(device=batch["clean_latents"].device, dtype=torch.float32)
        loss = (loss * weighting).mean()
        return loss

    def _try_remove_extra_checkpoints(self):
        if self.checkpointings_limit is None:
            return

        checkpoints = [
            d for d in os.listdir(self.output_dir) if d.startswith("checkpoint")
        ]

        if len(checkpoints) <= self.checkpointings_limit:
            return

        checkpoints = sorted(
            checkpoints,
            key=lambda x: self._parse_checkpoint_step(x),
            reverse=True,
        )

        for checkpoint in checkpoints[self.checkpointings_limit :]:
            checkpoint_path = os.path.join(self.output_dir, checkpoint)
            shutil.rmtree(checkpoint_path)
            self._info(f"Removed checkpoint {checkpoint_path}")

    def _save_checkpoint(self, global_step):
        self._accelerator.wait_for_everyone()

        if self._accelerator.is_main_process:
            self._try_remove_extra_checkpoints()

        if (
            self._accelerator.is_main_process
            or self._accelerator.distributed_type == DistributedType.DEEPSPEED
        ):
            save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
            self._accelerator.save_state(save_path)
            self._info(f"Saved checkpoint to {save_path}")

    def _final_save(self, transformer):
        if self._accelerator.is_main_process:
            self._save_model_hook([transformer], [], self.output_dir)
            self._info(f"Final model saved to {self.output_dir}")

    def _inspect_dtype(self, transformer):
        lora_layer: Any = transformer.transformer_blocks[0].attn.to_q  # type: ignore
        self._info(f"Diffusers dtype: {transformer.dtype}")
        self._info(f"Base layer dtype: {lora_layer.base_layer.weight.dtype}")
        self._info(f"LoRA layer dtype: {lora_layer.lora_A.default.weight.dtype}")

    def _move_batch_to_device(self, batch):
        new_batch = {}
        for k, v in batch.items():
            new_batch[k] = (
                v.to(self._accelerator.device) if isinstance(v, torch.Tensor) else v
            )
        return new_batch

    def _sample_and_log(self, transformer, global_step, progress):
        if self._sample_batch is None:
            return
        if self._accelerator.is_main_process:
            for key, batch in self._sample_batch.items():
                image = self.sampler.sample(
                    transformer,
                    self.adapter,
                    self._move_batch_to_device(batch),
                    progress=progress,
                )
                self._accelerator.log(
                    {f"sample/{key}": aim.Image(image)}, step=global_step
                )

    def train(self):
        set_seed(self.seed)

        transformer = self._make_transformer()
        self._accelerator.register_load_state_pre_hook(self._load_model_hook)
        self._accelerator.register_save_state_pre_hook(self._save_model_hook)
        self._optimize_model(transformer)

        optimizer = self._make_optimizer(transformer)
        dataloader = self._make_dataloader()
        self._calculate_real_train_steps(dataloader)
        lr_scheduler = self._make_lr_scheduler(optimizer)

        transformer, optimizer, dataloader, lr_scheduler = self._accelerator.prepare(
            transformer, optimizer, dataloader, lr_scheduler
        )

        global_step = self._try_resume_from_checkpoint()
        starting_epoch = global_step // len(dataloader)
        self._info(f"Starting training from epoch {starting_epoch}")

        if self._accelerator.is_main_process:
            if self._sample_batch is not None:
                self.sampler.load_model(
                    device=self._accelerator.device, dtype=self._weight_dtype
                )
                self._info(f"Loaded sampler model")
            progress = FluxFinetunerProgress()
            progress.start()

            self._sample_and_log(transformer, global_step, progress)

            task = progress.add_task(
                description="[bold blue]Training",
                total=self._train_steps,
                completed=global_step,
                epoch=starting_epoch,
                loss=0,
                lr=0,
                progress_type="train",
            )

        for epoch in range(starting_epoch, self._train_epochs):
            transformer.train()
            for step, batch in enumerate(dataloader):
                if global_step > self._train_steps:
                    break

                with self._accelerator.accumulate(transformer):
                    loss = self._train_step(transformer, batch)
                    self._accelerator.backward(loss)

                    if self._accelerator.sync_gradients:
                        self._accelerator.clip_grad_norm_(
                            transformer.parameters(), self.max_grad_norm
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                self._accelerator.log(logs, step=global_step)

                if self._accelerator.is_main_process and (
                    global_step % self.sample_steps == 0 or self.sample_steps == 1
                ):
                    self._sample_and_log(transformer, global_step, progress)
                    self._info(f"Sampled at step {global_step}")

                if (
                    global_step % self.checkpointing_steps == 0
                    or self.checkpointing_steps == 1
                ):
                    self._save_checkpoint(global_step)
                    self._info(f"Checkpoint saved at step {global_step}")

                if self._accelerator.is_main_process:
                    progress.update(task, advance=1, epoch=epoch, **logs)

        if self._accelerator.is_main_process:
            progress.stop()

        self._final_save(transformer)
        self._info("Training finished")
        self._accelerator.end_training()
