from transformers import Seq2SeqTrainer
from dataclasses import dataclass, field
from typing import Optional
import os
import datasets

class BartTraining(Seq2SeqTrainer):
    def __init__(self, model, dataloader, validation_loader, epochs, loss, optimizer, config, outputManager):
        dataset = dataloader.dataset
        valid_dataset = validation_loader.dataset

        super().__init__(model=model.model, 
            args=dataloader.args, train_dataset=dataset, eval_dataset=valid_dataset)

    def train(self):
        super().train()