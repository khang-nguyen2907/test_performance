import os
import sys
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("src") + len("src")])

import numpy as np
from transformers import (
    set_seed, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments
)
import evaluate 

from configs import parse_args
from data import CnnDailyMail
from utils import (
    postprocess_text, 
    CustomTrainingArguments, 
    CustomOptimizerTrainer, 
    detect_last_checkpoint
)
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# Metric
metric = evaluate.load("rouge")

def training_function(args):
    # set seed
    set_seed(args.seed)

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )

    # Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load dataset from disk and tokenizer
    dataset = CnnDailyMail(
        tokenizer=tokenizer, 
        dataset_id=args.dataset_id,
        dataset_config=args.data_config,
        tokenized_dataset_folder=args.tokenized_data_folder, 
        max_seq_length=args.max_seq_length, 
        max_label_length=args.max_label_length
    )
    train_dataset, eval_dataset = dataset.get_dataset()

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Initialize TrainingArguments 
    training_args = CustomTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train, 
        do_eval=args.do_eval,
        dataloader_num_workers=args.dataloader_num_workers,
        adafactor=args.adafactor,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.max_length,
        generation_num_beams=args.num_beams,
        fp16=args.fp16,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        optim=args.optim,
        final_cosine=args.final_cosine, 
        learning_rate=args.learning_rate,
        warmup_steps=args.num_warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        torch_compile=args.torch_compile,
        # logging & evaluation strategies
        logging_dir=args.logging_dir,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        # push to hub parameters
        eval_accumulation_steps=args.eval_accumulation_steps,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_strategy=args.hub_strategy,
        hub_model_id=args.repository_id,
        hub_token=args.hub_token,
    )

    # Initialize Trainer
    trainer = CustomOptimizerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Detect the last checkpoint    
    last_checkpoint = detect_last_checkpoint(training_args)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    kwargs = {"finetuned_from": args.model_name_or_path, "tasks": "up train GQA"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def main():
    args, _ = parse_args()
    training_function(args)