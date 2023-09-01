from .base.base_trainer import BaseTrainer

import logging 
import os
import math
from typing import (
    Optional, 
    Any, 
)

import datasets 
import torch 
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING, 
    MODEL_MAPPING,
    AutoConfig, 
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_scheduler, 
    Trainer, 
    DataCollatorForSeq2Seq
)
from transformers.trainer_utils import get_last_checkpoint

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class ProjectTrainerNoAPI(BaseTrainer):
    """ Training loop (this class is still not availabel yet)

    This class can be use immediately
    if there are some changed such as you want to use 
    other optimizer, scheduler, load different models, ... 
    you can override corresponding function in this class
    """
    def __init__(
            self, 
            model_args, 
            training_args, 
            data_args, 
            tokenizer: Optional[AutoTokenizer]=None, 
            model: Optional[AutoModelForSeq2SeqLM]=None
    ) -> None:
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        accelerator_log_kwargs = self._get_accelerator_log_kwargs()
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps, 
            **accelerator_log_kwargs
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process: 
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else: 
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
        
        # If passed along, set the training seed now
        if self.args.seed is not None: 
            set_seed(self.args.seed)

        self.model = model if model is not None else self._get_model()
        self.tokenizer = tokenizer if tokenizer is not None else self._get_tokenizer()

    def _get_model_config(self):
        """Get model's configuration"""
        if self.model_args.config_name: 
            config = AutoConfig.from_pretrained(self.model_args.config_name)
        elif self.model_args.model_name_or_path: 
            config = AutoConfig.from_pretrained(self.model_args.model_name_or_path)
        else: 
            config = CONFIG_MAPPING[self.model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch")
        
        return config
    
    def _get_tokenizer(self):
        """Get model's tokenizer"""
        if self.model_args.tokenizer_name: 
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.tokenizer_name, 
                use_fast=not self.model_args.use_slow_tokenizer)
        elif self.model_args.model_name_or_path: 
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.model_name_or_path, 
                use_fast=not self.model_args.use_slow_tokenizer
            )
        else: 
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using cfg.tokenizer_name"
            )
        return tokenizer
    
    def _get_model(self, config: AutoConfig = None):
        """Get model from public hub or local"""
        if config is None: 
            config = self._get_model_config()

        if self.model_args.use_decoder_only:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_args.model_name_or_path, 
                low_cpu_mem_usage=self.model_args.low_cpu_mem_usage
            )
        else: 
            logger.info("Traning new encoder-decoder model from scratch")
            model = AutoModelForSeq2SeqLM.from_config(config)
        
        return model
    
    def _get_optimizer(self):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.training_args.no_decay)],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.training_args.no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=self.training_args.lr
        )
        
        return optimizer
    
    def _get_scheduler(self):
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.training_args.gradient_accumulation_steps)
        if self.training_args.max_steps is None:
            self.training_args.max_steps = self.training_args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.training_args.warmup_steps * self.training_args.gradient_accumulation_steps,
            num_training_steps=self.training_args.max_steps * self.training_args.gradient_accumulation_steps,
        )

        return lr_scheduler, overrode_max_train_steps
    
    def _get_collate_fn(self): 
        label_pad_token_id = -100 if self.training_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if self.accelerator.use_fp16 else None,
        )
        return data_collator
    
    def train(self):
        # only show the progress bar once on each machine
        progress_bar = tqdm(
            range(self.training_args.max_steps), 
            disable=not self.accelerator.is_local_main_process
        )
        completed_steps = 0
        starting_epoch = 0
        resume_step = None
        # Potentially load in the weights and states from a previous save
        if self.training_args.resume_from_checkoint: 
            starting_epoch, completed_steps, resume_step = self._resume_from_checkpoint()
        
        progress_bar.update(completed_steps)

        for epoch in range(starting_epoch, self.training_args.num_train_epochs): 
            self.model.train()
            if self.training_args.with_tracking: 
                total_loss = 0
            if self.training_args.resume_from_checkoint and epoch == starting_epoch and resume_step is not None: 
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, resume_step)
            else: 
                active_dataloader = self.train_dataloader
            
            for step, batch in enumerate(active_dataloader): 
                with self.accelerator.accumulate(self.model): 
                    loss = self.model(batch)

                    # We keep track of the loss at each epoch 
                    if self.training_args.with_tracking: 
                        total_loss += loss
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients: 
                    progress_bar.update(1)
                    completed_steps += 1
                
                self.save_checkpoint(completed_steps)

                if completed_steps >= self.training_args.max_steps:
                    break

    def save_checkpoint(
            self, 
            completed_steps: int) -> None: 
        if isinstance(self.checkpointing_steps, int): 
            if completed_steps % self.checkpointing_steps == 0: 
                output_dir = f"step_{completed_steps}"
                if self.training_args.output_dir is not None: 
                    output_dir = os.path.join(self.training_args.output_dir, output_dir)
                self.accelerator.save_state(output_dir)
    
    def _resume_from_checkpoint(self):
        if self.training_args.resume_from_checkoint is not None or self.training_args.resume_from_checkoint != "":
            self.accelerator.print(f"Resumed from checkpointed: {self.training_args.resume_from_checkoint}") 
            self.accelerator.load_state(self.training_args.resume_from_checkoint)
            path = os.path.basename(self.training_args.resume_from_checkoint)
        else: 
            # Get the most recent checkpoint 
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(ket=os.path.getctime)
            path=dirs[-1] # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splittext(path)[0]

        if "epoch" in training_difference: 
            starting_epoch = int(training_difference.replace("step_","")) + 1
            resume_step = None
            completed_steps = starting_epoch*self.num_update_steps_per_epoch
        else: 
            # need to multiply `gradient_accumulation_steps` to reflect real steps 
            resume_step = int(training_difference.replace("step_", "")) * self.training_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(self.train_dataloader)
            resume_step -= starting_epoch * len(self.train_dataloader)
            completed_steps = resume_step // self.training_args.gradient_accumulation_steps
        return starting_epoch, completed_steps, resume_step

    def evaluate(self):
        return super().evaluate()

class ProjectTrainerAPI(BaseTrainer):
    """ Trainer API HuggingFace 

    This class can be use immediately
    if there are some changed such as you want to use 
    other optimizer, scheduler, load different models, ... 
    you can override corresponding function in this class
    """
    def __init__(
            self, 
            model_args, 
            training_args, 
            data_args, 
            train_dataset, 
            eval_dataset, 
            tokenizer: Optional[AutoTokenizer]=None, 
            model: Optional[AutoModelForSeq2SeqLM]=None, 
            optimizer: Optional[Any] = None, 
            scheduler: Optional[Any] = None, 
            collate_fn: Optional[Any] = None
    ) -> None:
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args

        self.optimizer = optimizer if optimizer is not None else self._get_optimizer()
        self.scheduler = scheduler if scheduler is not None else self._get_scheduler()
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = model if model is not None else self._get_model()
        self.tokenizer = tokenizer if tokenizer is not None else self._get_tokenizer()
        self.collate_fn = collate_fn if collate_fn is not None else self._get_collate_fn()

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args, 
            data_collator=self.collate_fn, 
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.eval_dataset if self.training_args.do_eval else None, 
            optimizers=(self.optimizer, self.scheduler), 
            compute_metrics=self.compute_metric
        )

        # Detecting last checkpoint.
        self.last_checkpoint = None
        if os.path.isdir(self.raining_args.output_dir) and self.training_args.do_train and not self.training_args.overwrite_output_dir:
            self.last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if self.last_checkpoint is None and len(os.listdir(self.training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif self.last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {self.last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

    def _get_model_config(self):
        """Get model's configuration"""
        if self.model_args.config_name: 
            config = AutoConfig.from_pretrained(self.model_args.config_name)
        elif self.model_args.model_name_or_path: 
            config = AutoConfig.from_pretrained(self.model_args.model_name_or_path)
        else: 
            config = CONFIG_MAPPING[self.model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch")
        
        return config
    
    def _get_tokenizer(self):
        """Get model's tokenizer"""
        if self.model_args.tokenizer_name: 
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.tokenizer_name, 
                use_fast=not self.model_args.use_slow_tokenizer)
        elif self.model_args.model_name_or_path: 
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.model_name_or_path, 
                use_fast=not self.model_args.use_slow_tokenizer
            )
        else: 
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using cfg.tokenizer_name"
            )
        return tokenizer
    
    def _get_model(self, config: AutoConfig = None):
        """Get model from public hub or local"""
        if config is None: 
            config = self._get_model_config()

        if self.model_args.use_decoder_only:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_args.model_name_or_path, 
                low_cpu_mem_usage=self.model_args.low_cpu_mem_usage
            )
        else: 
            logger.info("Traning new encoder-decoder model from scratch")
            model = AutoModelForSeq2SeqLM.from_config(config)
        
        return model
    
    def _get_optimizer(self):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.training_args.no_decay)],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.training_args.no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=self.training_args.lr
        )
        
        return optimizer
    
    def _get_scheduler(self):
        # Scheduler and math around the number of training steps

        return None
    
    def _get_collate_fn(self): 
        label_pad_token_id = -100 if self.training_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )
        return data_collator
    
    def train(self):
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif self.last_checkpoint is not None:
            checkpoint = self.last_checkpoint
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(self.train_dataset)
        )

        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

    def save_checkpoint(
            self, 
            completed_steps: int) -> None: 
        if isinstance(self.checkpointing_steps, int): 
            if completed_steps % self.checkpointing_steps == 0: 
                output_dir = f"step_{completed_steps}"
                if self.training_args.output_dir is not None: 
                    output_dir = os.path.join(self.training_args.output_dir, output_dir)
                self.accelerator.save_state(output_dir)
    
    def _resume_from_checkpoint(self):
        return None

    def evaluate(self):
        results = {}
        logger.info("**** Evaluate ***")
        if isinstance(self.eval_dataset, dict): 
            metrics = {}
            for eval_ds_name, eval_ds in self.eval_dataset.items(): 
                dataset_metrics = self.trainer.evaluate(
                    eval_dataset=eval_ds, 
                    metric_key_prefix=f"eval_{eval_ds_name}"
                )
                metrics.update(dataset_metrics)
        
        else: 
            metrics = self.trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(self.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))
    
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def compute_metric(self):
        return super().compute_metric()