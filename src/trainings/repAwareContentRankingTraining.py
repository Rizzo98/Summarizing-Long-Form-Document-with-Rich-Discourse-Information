import wandb

class RepAwareContentRankingTraining:
    def __init__(self, model, dataloader, validation_loader, epochs, loss, optimizer, config, outputManager):
        self.model = model
        self.dataloader = dataloader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.config = config
        self.outputManager = outputManager

    def train(self):
        self.model.train()
        for e in range(self.epochs):
            epoch_loss = 0
            for batch, (data, sections_gold, sentences_gold) in enumerate(self.dataloader):
                sections_importance, sentences_importance, _ = self.model(data, device=self.config.device)
                loss = self.loss(sections_importance, sentences_importance, sections_gold, sentences_gold)
                epoch_loss+= loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            valid_loss = self.valid_evaluate()
            if self.config.logging:
                self.outputManager.writeLog(f'Epoch {e}/{self.epochs}. Training loss: {epoch_loss:.3f}. Valid loss: {valid_loss:.3f}')
            if self.config.wandb:
                wandb.log({"Training loss": epoch_loss, 'Valid loss': valid_loss})
            if self.config.save_model:
                self.outputManager.saveModel(self.model)
            print(f'Epoch {e}/{self.epochs}. Training loss: {epoch_loss:.3f}. Valid loss: {valid_loss:.3f}')

    def valid_evaluate(self):
        #Compute loss on eval set
        total_loss = 0
        for batch, (data, sections_gold, sentences_gold) in enumerate(self.validation_loader):
            sections_importance, sentences_importance, _ = self.model(data, device=self.config.device)
            total_loss += self.loss(sections_importance, sentences_importance, sections_gold, sentences_gold)
        return total_loss
        

