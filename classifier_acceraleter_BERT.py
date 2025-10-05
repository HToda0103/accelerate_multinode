import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from accelerate import Accelerator
from torch.utils.data.distributed import DistributedSampler

def main():
    # 1. Acceleratorの初期化
    accelerator = Accelerator()

    MODEL_NAME = 'tohoku-nlp/bert-base-japanese-whole-word-masking'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
        use_safetensors=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id


    try:
        df = pd.read_csv("all_text.tsv", delimiter='\t', header=None, names=['media_name', 'label', 'NaN', 'sentence'])
    except FileNotFoundError:
        if accelerator.is_main_process:
            print("Error: all_text.tsv not found. Please check the file path.")
        exit()

    df = df.dropna(subset=['sentence', 'label'])
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    sentences = df.sentence.values
    labels = df.label.values

    input_ids = []
    attention_masks = []
    MAX_LEN = 128

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if accelerator.is_main_process:
        print(f'Number of training data: {train_size}')
        print(f'Number of validation data: {val_size}')

    batch_size = 124

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )
    print(f'Number of batches in training dataloader: {len(train_dataloader)}')
    print(f'Number of batches in validation dataloader: {len(validation_dataloader)}')
    optimizer = AdamW(model.parameters(), lr=2e-5)
    print(f'Wrapping starts...')

    # 2. Prepare the model, optimizer, and dataloaders
    model, optimizer, train_dataloader, validation_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader
    )
    print(f'Device to use: {accelerator.device}')
    max_epoch = 1

    if accelerator.is_main_process:
        print("\nStarting training...")

    for epoch in range(max_epoch):
        print(f"Starting epoch {epoch + 1}")
        # --- Training loop ---
        model.train()
        total_train_loss = 0
        for i,batch in enumerate(train_dataloader):
            print(f'Processing batch {i + 1}/{len(train_dataloader)}...')
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                labels=batch[2]
            )
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Use accelerator.backward(loss)
            accelerator.backward(loss)
            
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- Validation loop ---
        model.eval()
        total_val_loss = 0
        if len(validation_dataloader) > 0:
            with torch.no_grad():
                for batch in validation_dataloader:
                    outputs = model(
                        input_ids=batch[0],
                        attention_mask=batch[1],
                        labels=batch[2]
                    )
                    loss = outputs.loss
                    total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(validation_dataloader) if len(validation_dataloader) > 0 else 0

        # Display logs only on the main process
        if accelerator.is_main_process:
            print(f'\nEpoch {epoch + 1}/{max_epoch}')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            if avg_val_loss > 0:
                print(f'  Valid Loss: {avg_val_loss:.4f}')

    if accelerator.is_main_process:
        print("\nTraining complete.")
    
    # The prediction part below needs adjustment depending on the implementation, 
    # but it is a basic conversion example.
    # To get accurate accuracy, you need to aggregate the results of all processes with accelerator.gather().
    if val_size > 0 and accelerator.is_main_process:
        model.eval()
        print("\nRunning predictions on validation data...")
        for batch in validation_dataloader:
            with torch.no_grad():
                outputs = model(
                    input_ids=batch[0],
                    attention_mask=batch[1]
                )
            
            logits = outputs.logits
            logits_df = pd.DataFrame(logits.cpu().numpy(), columns=['logit_0', 'logit_1'])
            pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
            pred_df = pd.DataFrame(pred_labels, columns=['pred_label'])
            label_df = pd.DataFrame(batch[2].cpu().numpy(), columns=['true_label'])
            accuracy_df = pd.concat([logits_df, pred_df, label_df], axis=1)
            print("\nSample prediction results for a validation batch:")
            print(accuracy_df.head())
            accuracy = (accuracy_df['pred_label'] == accuracy_df['true_label']).mean()
            print(f"\nAccuracy for this batch: {accuracy:.4f}")
            break # 最初のバッチのみで評価

if __name__ == '__main__':
    main()
