import os
import sys
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from transformers import (
    MODEL_MAPPING, 
)
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("src") + len("src")])

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import argparse

def parse_args(): 
    parser = argparse.ArgumentParser(
        description="T5 Summarization"
    )

    # Model Arguments

    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="google/flan-t5-large", 
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )

    parser.add_argument(
        "--config_name", 
        type=str, 
        default=None, 
        help="Pretrained config name or path if not the same as model_name"
    )

    parser.add_argument(
        "--tokenizer_name", 
        type=str, 
        default=None, 
        help="Pretrained tokenizer name or path if not the same as model_name"
    )

    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None, 
        help="Path to directory to store the pretrained models downloaded from huggingface.co"
    )

    parser.add_argument(
        "--use_fast_tokenizer", 
        action="store_false", 
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
    )

    parser.add_argument(
        "--use_auth_token", 
        action="store_true",
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
    )

    # Data Arguments 

    parser.add_argument(
        "--tokenized_data_folder", 
        type=str, 
        default="./data/local/vit5/raw_dataset_T5_512/tokenized_raw_dataset_T5_512", 
        help="The path to the tokenized dataset folder."
    )


    parser.add_argument(
        "--dataset_id", 
        type=str, 
        default="cnn_dailymail", 
        help="Hugging Face Model Id"
    )

    parser.add_argument(
        "--dataset_config", 
        type=str, 
        default="3.0.0", 
        help="config/verison of the dataset"
    )

    parser.add_argument(
        "--preprocessing_num_workers", 
        type=int, 
        default=2, 
        help="The number of processes to use for the preprocessing."
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default='article',
        help="column of input text is"
    )

    parser.add_argument(
        "--summary_column",
        type=str,
        default='highlights',
        help="The column name handling"
    )

    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=0,
        help="The index to start make sentinel"
    )

    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=512, 
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        )
    )

    parser.add_argument(
        "--max_label_length",
        type=int,
        default=None,
        help=(
            "label length"
        )
    )

    parser.add_argument(
        "--max_answer_length", 
        type=int, 
        default=168, 
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        )
    )


    parser.add_argument(
        "--pad_to_max_length", 
        action="store_false",
        help=(
            "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
            " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
        )
    )

    # Generation argument

    parser.add_argument(
        "--num_beams", 
        type=int, 
        default=1, 
        help=(
            "Number of beams for beam search. 1 means no beam search."
        )
    )

    parser.add_argument(
        "--penalty_alpha", 
        type=float, 
        default=None, 
        help=(
            "The values balance the model confidence and the degeneration penalty in contrastive search decoding."
        )
    )

    parser.add_argument(
        "--use_cache", 
        action="store_true",
        help=(
            "Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding."
        )
    )

    parser.add_argument(
        "--top_k", 
        type=int, 
        default=25, 
        help=(
            "The number of highest probability vocabulary tokens to keep for top-k-filtering."
        )
    )

    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help=(
            "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."
        )
    )

    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help=(
            "The value used to modulate the next token probabilities."
        )
    )

    parser.add_argument(
        "--do_sample", 
        action="store_true",
        help=(
            "Whether or not to use sampling ; use greedy decoding otherwise."
        )
    )

    parser.add_argument(
        "--max_length", 
        type=int, 
        default=150, 
        help=(
            "The maximum length the generated tokens can have."
        )
    )

    parser.add_argument(
        "--min_length", 
        type=int, 
        default=150, 
        help=(
            "The minimum length of the sequence to be generated."
        )
    )

    parser.add_argument(
        "--num_return_sequences", 
        type=int, 
        default=1, 
        help=(
            "The number of independently computed returned sequences for each element in the batch."
        )
    )

    parser.add_argument(
        "--no_repeat_gram_size", 
        type=int, 
        default=3, 
        help=(
            "If set to int > 0, all ngrams of that size can only occur once."
        )
    )

    parser.add_argument(
        "--repetition_penalty", 
        type=float, 
        default=1.1, 
        help=(
            "The parameter for repetition penalty. 1.0 means no penalty."
        )
    )

    parser.add_argument(
        "--ignore_pad_token_for_loss", 
        action="store_true",
        help=(
            "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        )
    )
    
    # Training Arguments 

    parser.add_argument(
        "--do_train", 
        action="store_true",
        help=(
            "whether to train or not"
        )
    )

    parser.add_argument(
        "--do_eval", 
        action="store_true",
        help=(
            "whether to evaluate or not"
        )
    )

    parser.add_argument(
        "--overwrite_output_dir", 
        action="store_true",
        help=(
            "reuse output dir"
        )
    )

    parser.add_argument(
        "--output_dir", 
        type=str,
        default="./model/t5",
        help=(
            "where to save the checkpoints"
        )
    )

    parser.add_argument(
        "--num_groups", 
        type=int, 
        default=8, 
        help=(
            "Number of group in multi-group attention."
        )
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help=(
            "A seed for reproducible training."
        )
    )

    parser.add_argument(
        "--model_type", 
        type=int, 
        default=8, 
        help=(
            "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        ), 
        choices=MODEL_TYPES
    )

    parser.add_argument(
        "--checkpointing_steps", 
        type=int, 
        default=8, 
        help=(
            "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        ), 
    )

    parser.add_argument(
        "--with_tracking", 
        action="store_true",
        help=(
            "Whether to enable experiment trackers for logging."
        )
    )

    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=8, 
        help=(
            "Batch size (per device) for the training dataloader."
        ), 
    )

    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=8, 
        help=(
            "Batch size (per device) for the evaluation dataloader."
        ), 
    )

    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4, 
        help=(
            "Batch size (per device) for the evaluation dataloader."
        ), 
    )

    parser.add_argument(
        "--adafactor", 
        action="store_true", 
        help=(
            "whether to use adafactor or not"
        ), 
    )

    parser.add_argument(
        "--optim", 
        type=str, 
        default="adamwscale", 
        help=(
            "`adamw`, `adamwscale`, `adafactor`"
        ),
    )

    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=10, 
        help=(
            "Total number of training epochs to perform."
        ), 
    )

    parser.add_argument(
        "--num_update_steps_per_epoch", 
        type=int,
        default=-1,
        help=(
            "Total number of update steps."
        )
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs."
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Model's checkpoints will be saved after `save_steps` steps"
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Save GPU memory usage."
    )

    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether or not to compile the model using PyTorch 2.0 torch.compile."
    )

    parser.add_argument(
        "--deepspeed",
        type=str,
        default="./src/configs/deepspeed_config/ds_t5_z3_offload.json",
        help="Path to deepspeed config .json file."
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="The scheduler type to use."
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--final_cosine",
        type=float,
        default=1e-5,
        help="final cosine"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        default=16,
        help="Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but requires more memory)"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder."
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to enable fp16."
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to enable bfloat16."
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["all", "tensorboard", "wandb", "comet_ml", "clearml"],
        help='The integration to report the results and logs to. Supported platforms are "tensorboard", "wandb", "comet_ml", and "clearml". Use "all" (default) to report to all integrations. Only applicable when `--with_tracking` is passed.'
    )


    parser.add_argument(
        "--predict_with_generate",
        action="store_true",
        help="Generate text to evaluate the model's performance."
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the checkpoint model to hub. It must be provided repository id."
    )

    parser.add_argument(
        "--hub_strategy",
        type=str,
        default="every_save",
        help="Push-to-hub strategy."
    )

    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Token to use for uploading models to Hugging Face Hub. Hub token for being private."
    )

    parser.add_argument(
        "--repository_id",
        type=str,
        default=None,
        help="Hugging Face Repository id for uploading models."
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="where to save training log"
    )

    parser.add_argument(
        "--logging_strategy",
        type=str,
        default="steps",
        help="Log training info after epochs or steps"
    )

    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="loss",
        help=" Use in conjunction with load_best_model_at_end to specify the metric to use to compare two different models. Must be the name of a metric returned by the evaluation with or without the prefix `eval_`. Will default to `loss` if unspecified and load_best_model_at_end=True (to use the evaluation loss)."
    )
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="How many steps to log training info"
    )

    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        help="Evaluate model after `epoch` or `steps`, if not `no`, `do_eval` will be automatically set to True. "
    )

    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        help="Save model's checkpoint after steps or epochs."
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="How many saved checkpoints could be existing in `output_dir`."
    )

    parser.add_argument(
        "--load_best_model_at_end",
        type=bool,
        default=False,
        help="Always load best model/checkpoint."
    )
    args = parser.parse_known_args()

    return args