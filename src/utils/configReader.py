import json
from src.models import *
from src.trainings import *
from src.losses import *
import torch
import shutil
import os
import copy

class ModelConfig:
    def __init__(self,model, device):
        self.modelClass, modelConfig = ConfigReader.modelMapper(model)
        self.do_train = model['train']
        self.pretrained_model = model['pretrained_model']
        self.do_inference = model['inference']
        self.from_previous = model['from_previous']
        self.tokenizer = ConfigReader.tokenizerMapped(modelConfig['tokenizer'])
        self.modelParams = ConfigReader.modelParamsMapped(modelConfig['params'])
        self.training = {
            'training_class': ConfigReader.trainingMapped(modelConfig['train']['training_class']),
            'dataset': ConfigReader.datasetMapped(modelConfig['train']['training_dataset'],self.modelClass,self.tokenizer),
            'validation': ConfigReader.datasetMapped(modelConfig['train']['validation_dataset'],self.modelClass,self.tokenizer),
            'dataloader': ConfigReader.dataloaderMapped(modelConfig['train']['training_dataloader'],device),
            'epochs': modelConfig['train']['training_epochs'],
            'optimizer': ConfigReader.optimizerMapper(modelConfig['train']['optimizer']),
            'loss': ConfigReader.lossMapped(modelConfig['train']['loss'])
        }
        self.inference = ConfigReader.inferenceMapped(modelConfig['inference'],self.modelClass, self.training['dataset'], self.training['dataloader'])
    
    def get_dict(self):
        d = {
            'name': self.modelClass.__name__,
            'pretrained': self.pretrained_model,
            'tokenizer': self.tokenizer['class'].__name__,
            'params': self.modelParams,
            'training_set': self.training['dataset']['params']['data_path'],
            'validation_set': self.training['validation']['params']['data_path'],
            'epochs': self.training['epochs'],
            'optimizer': self.training['optimizer']['class'].__name__ if self.training['optimizer'] != None else None ,
            'optimizer_params': self.training['optimizer']['params'] if self.training['optimizer'] != None else None,
            'loss': self.training['loss']['class'].__name__ if self.training['loss'] != None else None
        }
        return d

class ConfigReader:
    def __init__(self): 
        with open('./config/config.json','r') as f:
            self.__config = json.load(f)
        f.close()
        self.device = self.__config['device']
        self.models = [ModelConfig(model,self.device) for model in self.__config['models']]
        self.__outputSetup(self.__config['output'])

    def getConfigDict(self):
        total_config = copy.deepcopy(self.__config)
        total_config['models'] = []
        for m in self.models:
            total_config['models'].append(m.get_dict())
        return total_config

    @staticmethod
    def modelMapper(model,config=True):
        models = {
            'ContentRanking': ContentRanking,
            'Bart': Bart,
            'RepAwareContentRanking': RepAwareContentRanking
        }
        m = models[model['name']]
        if config:
            model_config = json.load(open(f'./config/{model["config"]}','r'))
            return m, model_config
        else:
            return m

    @staticmethod
    def optimizerMapper(opt):
        if opt['name']=='default':return
        optimizers = {
            'SGD': torch.optim.SGD
        }
        to_return = {
            'params':dict([(k,v) for k,v in opt.items() if k!='name'])
        }
        to_return['class'] = optimizers[opt['name']]
        return to_return

    @staticmethod
    def lossMapped(data):
        if data['name']=='default':return
        losses = {
            'ContentRankingLoss': ContentRankingLoss
        }
        to_return = {
            'params': dict([(k,v) for k,v in data.items() if k!='name'])
        }
        to_return['class'] = losses[data['name']]
        return to_return

    @staticmethod
    def trainingMapped(data):
        trainings = {
            'ContentRankingTraining': ContentRankingTraining,
            'BartTraining': BartTraining,
            'RepAwareContentRankingTraining': RepAwareContentRankingTraining
        }
        return trainings[data]

    @staticmethod
    def datasetMapped(data, fromClass, tokenizer):
        from src.datasets import DocumentDataset, BartDataset, DynamicImportanceContentRankingDataset
        datasets = {
            'DocumentDataset': DocumentDataset,
            'BartDataset': BartDataset,
            'DynamicImportanceContentRankingDataset': DynamicImportanceContentRankingDataset
        }
        to_return = {
            'params': dict([(k,v) for k,v in data.items() if k!='class' and k!='tokenizer'])
        }
        if tokenizer!=None:
            to_return['params']['tokenizer'] = tokenizer

        to_return['class'] = datasets[data['class']]

        return to_return

    @staticmethod
    def dataloaderMapped(data, device):
        from src.dataloaders import BartDataLoader, ContentRankingDataLoader

        dataloaders = {
            'ContentRankingDataLoader': ContentRankingDataLoader,
            'BartDataLoader': BartDataLoader
        }
        to_return = dict([(k,v) for k,v in data.items() if k!='class'])

        to_return['class'] = dataloaders[data['class']]
        to_return['params']['device'] = device
        return to_return
    
    @staticmethod
    def tokenizerMapped(data):
        from src.tokenizers import BartTokenizer, ContentRankingTokenizer

        if data==None: return
        tokenizers = {
            'ContentRankingTokenizer': ContentRankingTokenizer,
            'BartTokenizer': BartTokenizer
        }
        to_return = dict([(k,v) for k,v in data.items() if k!='class'])
        to_return['class'] = tokenizers[data['class']]
        return to_return
    
    @staticmethod
    def modelParamsMapped(data):
        to_return = {
            'params':dict([(k,v) for k,v in data.items() if k!='tokenizer'])
        }
        to_return['tokenizer'] = data['tokenizer']
        return to_return


    @staticmethod
    def inferenceMapped(data, fromClass, dataset, dataloader):
        from src.utils import Inference

        to_return = dict([(k,v) for k,v in data.items()])
        to_return['method'] = Inference.inferenceFrom(fromClass)
        to_return['dataset'] = dataset
        to_return['dataset']['params']['data_path'] = data['data_path']
        to_return['dataloader'] = copy.deepcopy(dataloader)
        to_return['dataloader']['params']['batch_size'] = 1
        to_return['dataloader']['params']['shuffle'] = False
        return to_return


    def __outputSetup(self,data):
        self.output_folder = f'./outputs/{data["folder"]}'
        if os.path.isdir(self.output_folder):
            print("Overwrite output folder")
            shutil.rmtree(self.output_folder)
        os.mkdir(self.output_folder)
        self.save_model = data['save_model']
        self.logging = data['logging']
        self.wandb = data['wandb']
            