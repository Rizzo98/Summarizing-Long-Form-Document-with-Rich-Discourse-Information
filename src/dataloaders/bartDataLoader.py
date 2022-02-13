from transformers import TrainingArguments, Seq2SeqTrainingArguments

class BartDataLoader():
    def __init__(self, dataset, batch_size=32, encoder_max_length=100, decoder_max_length=20,
     num_train_epochs=3, device='cpu', shuffle=False, outputFolder=''):
        self.tokenizer = dataset.tokenizer
        self.raw_docs = dataset.raw_docs
        self.dataset = dataset().map( 
            lambda x:x,
            batched=True, 
            batch_size=batch_size
        )
        self.dataset.set_format(
            type="torch", columns=['Doc_id', "input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )
        self.args = Seq2SeqTrainingArguments(
            output_dir=f"./outputs/{outputFolder}/",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            predict_with_generate=True,
            evaluation_strategy="epoch",
            do_train=True,
            do_eval=True,
            logging_steps=2, 
            save_steps=16, 
            eval_steps=500, 
            warmup_steps=500, 
            overwrite_output_dir=True,
            save_total_limit=1,
            report_to='none'
            )
        
    def __iter__(self):
        yield tuple(self.dataset[0].values())
    
    def __len__(self):
        return len(self.dataset)