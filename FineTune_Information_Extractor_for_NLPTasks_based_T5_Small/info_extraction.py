import os
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import gc
from sklearn.model_selection import train_test_split

# Set CUDA launch blocking for detailed error tracking
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class InfoExtractionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, task_instruction, max_length=1024):
        self.inputs = []
        self.labels = []
        
        for text, label in zip(texts, labels):
            input_text = f"{task_instruction}: {text}"
            inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
            labels = tokenizer(label, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").input_ids
            self.inputs.append(inputs['input_ids'].squeeze(0))
            self.labels.append(labels.squeeze(0))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input_ids': self.inputs[idx], 'labels': self.labels[idx]}

class InfoExtractionModel:
    def __init__(self, model_name='t5-small', cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.getcwd()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device)

    def train(self, texts_train, labels_train, texts_eval, labels_eval, task_instruction, num_epochs, output_dir='./info_extraction_model'):
        print(f"Training the model for task: {task_instruction} for {num_epochs} epoch(s)")
        train_dataset = InfoExtractionDataset(texts_train, labels_train, self.tokenizer, task_instruction)
        eval_dataset = InfoExtractionDataset(texts_eval, labels_eval, self.tokenizer, task_instruction)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,  # Use the user-defined number of epochs
            per_device_train_batch_size=1,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy="epoch",
            logging_dir='./logs',
            learning_rate=5e-5,  
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
        self.save_fine_tuned_model(output_dir)
        print(f"Model trained and saved to {output_dir}")

        del train_dataset, eval_dataset, trainer, training_args
        torch.cuda.empty_cache()
        gc.collect()

    def save_fine_tuned_model(self, output_dir):
        """Method to save the fine-tuned model and tokenizer."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.model.save_pretrained(output_dir)  # Save the fine-tuned model
        self.tokenizer.save_pretrained(output_dir)  # Save the tokenizer
        print(f"Fine-tuned model and tokenizer saved to {output_dir}")


    def load_model(self, model_dir):
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        print(f"Model loaded from {model_dir}")

    def extract(self, text, task_instruction):
        print(f"Extracting information based on task: {task_instruction}")
        input_text = f"{task_instruction}: {text}"
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        extracted_info = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        print(f"Extracted information: {extracted_info}")
        
        del inputs, output_ids
        torch.cuda.empty_cache()
        gc.collect()

        return extracted_info

    def save_model(self, output_dir):
        """Saves the fine-tuned model and tokenizer to the specified directory."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned model and tokenizer saved to {output_dir}")


