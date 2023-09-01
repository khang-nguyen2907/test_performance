from typing import Optional, Tuple
from datasets import (
    load_dataset, 
    concatenate_datasets, 
    DatasetDict, 
    Dataset, 
    load_from_disk
)
import numpy as np 
import os 

class CnnDailyMail: 
    def __init__(
            self,
            tokenizer, 
            dataset_id: Optional[str]=None, 
            dataset_config: Optional[str]=None, 
            tokenized_dataset_folder: Optional[str]=None, 
            max_seq_length: int = 512, 
            max_label_length: int = 129,
    ) -> None:
        tokenized_dataset_folder_exist = os.path.isdir(tokenized_dataset_folder)
        if (
            dataset_id is None
            and dataset_config is None
            and not tokenized_dataset_folder_exist
        ): 
            raise ValueError("dataset information must be provided")
        
        self.tokenizer = tokenizer
        self.dataset_id = dataset_id
        self.dataset_config = dataset_config
        self.tokenized_dataset_folder = tokenized_dataset_folder

        if not tokenized_dataset_folder_exist: 
            self.text_dataset = self.download()
            self.train_dataset, self.eval_dataset = self.tokenize(max_seq_length=max_seq_length, max_label_length=max_label_length)
            
        else: 
            self.train_dataset = load_from_disk(os.path.join(self.tokenized_dataset_folder, "train"))
            self.eval_dataset = load_from_disk(os.path.join(self.tokenized_dataset_folder, "test"))
    
    def get_dataset(self) -> Tuple[Dataset]: 
        """Get train, evaluation dataset"""
        return self.train_dataset, self.eval_dataset
    
    def download(self) -> DatasetDict:
        """Download dataset from hub"""
        dataset = load_dataset(self.dataset_id,name=self.dataset_config)
        return dataset

    def tokenize(
            self, 
            max_seq_length: int = 512, 
            max_label_length: int = 129
        ) -> Tuple[Dataset]: 
        """Tokenize text dataset
        
        Args: 
            max_seq_length: max input sequence length 
            max_label_length: max label sequence length
        """
        if self.text_dataset is None:
            raise ValueError("Text Dataset is not created correctly")
        
        prompt_template = f"Summarize the following news article:\n{{input}}\nSummary:\n"
        text_column = "article" # column of input text is
        summary_column = "highlights" # column of the output text 

        prompt_length = len(self.tokenizer(prompt_template.format(input=""))["input_ids"])
        max_sample_length = self.tokenizer.model_max_length - prompt_length
        # The maximum total input sequence length after tokenization. 
        # Sequences longer than this will be truncated, sequences shorter will be padded.
        tokenized_inputs = concatenate_datasets([self.text_dataset["train"], self.text_dataset["test"]]).map(lambda x: self.tokenizer(x[text_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
        max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
        max_source_length = min(max_source_length, max_sample_length)
        print(f"Max source length: {max_source_length}")

        # The maximum total sequence length for target text after tokenization. 
        # Sequences longer than this will be truncated, sequences shorter will be padded."
        tokenized_targets = concatenate_datasets([self.text_dataset["train"], self.text_dataset["test"]]).map(lambda x: self.tokenizer(x[summary_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
        target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
        # use 95th percentile as max target length
        if max_label_length > 0:
            max_target_length = max_label_length
        else:
            max_target_length = int(np.percentile(target_lenghts, 95))

        print(f"Max target length: {max_target_length}")

        def preprocess_function(sample, padding="max_length"):
            # created prompted input
            inputs = [prompt_template.format(input=item) for item in sample[text_column]]

            # tokenize inputs
            model_inputs = self.tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)

            # Tokenize targets with the `text_target` keyword argument
            labels = self.tokenizer(text_target=sample[summary_column], max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # process dataset
        tokenized_dataset = self.text_dataset.map(preprocess_function, batched=True, remove_columns=list(self.dataset["train"].features))

        # save dataset to disk
        tokenized_dataset["train"].save_to_disk(os.path.join(self.tokenized_dataset_folder,"train"))
        tokenized_dataset["test"].save_to_disk(os.path.join(self.tokenized_dataset_folder,"eval"))
        return tokenized_dataset["train"], tokenized_dataset["test"]