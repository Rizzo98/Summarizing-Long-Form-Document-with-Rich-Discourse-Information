from src.utils import ConfigReader, ModelConfig
from src.utils import OutputManager
from src.utils import Wrapper
import torch
import wandb
import sys
import copy

'''
TODO:
-big refactory of the code
-write README
-merge branches on github
-compute rouge on results
'''

sys.setrecursionlimit(10000)
config = ConfigReader()

outputManager = OutputManager(config.output_folder)
if config.wandb:
    wandb.init(
        project="Summarizing-Long-Form-Document-with-Rich-Discourse-Information", 
        entity="riz98",
        config = {})
    wandb.config.update(config.getConfigDict())

if config.logging:
    outputManager.writeLog(config.getConfigDict())


for modelIndex in range(len(config.models)):
    modelConfig = config.models[modelIndex]

    if modelConfig.from_previous:
        assert modelIndex>0, 'First model cannot have wrapper!'
        standard = config.models[modelIndex-1].inference['method'](model,dataloader,config.device,**config.models[modelIndex-1].inference['params'])
        dataset = Wrapper.wrapperTo(modelConfig.modelClass)(standard,modelConfig.training['dataset']['params'])

        standard = config.models[modelIndex-1].inference['method'](model,validation_loader,config.device,**config.models[modelIndex-1].inference['params'])
        valid_params = copy.deepcopy(modelConfig.training['validation']['params'])
        valid_params.update({'validation_set':True, 'pre_trained_tokenizer':dataset.tokenizer})
        validationset = Wrapper.wrapperTo(modelConfig.modelClass)(standard,valid_params)
        
        if modelConfig.do_inference:
            standard = config.models[modelIndex-1].inference['method'](model,inference_loader,config.device,**config.models[modelIndex-1].inference['params'])
            infer_params = copy.deepcopy(modelConfig.training['dataset']['params'])
            infer_params.update({'validation_set':True, 'pre_trained_tokenizer':dataset.tokenizer})
            inferenceset = Wrapper.wrapperTo(modelConfig.modelClass)(standard,infer_params)
    else:
        dataset = modelConfig.training['dataset']['class'](**modelConfig.training['dataset']['params']) 

        validationset = modelConfig.training['validation']['class'](**modelConfig.training['validation']['params'],
            validation_set=True, pre_trained_tokenizer = dataset.tokenizer)

    dataloader = modelConfig.training['dataloader']['class'](dataset,**modelConfig.training['dataloader']['params'])
    validation_loader = modelConfig.training['dataloader']['class'](validationset,**modelConfig.training['dataloader']['params'])

    if modelConfig.modelParams['tokenizer']:
        model = modelConfig.modelClass(tokenizer=dataset.tokenizer,**modelConfig.modelParams['params'], device=config.device)
    else:
        model = modelConfig.modelClass(**modelConfig.modelParams['params'], device=config.device)
    
    if modelConfig.do_train:
        if modelConfig.training['loss']!=None:     
            loss = modelConfig.training['loss']['class'](**modelConfig.training['loss']['params'],device=config.device)
        else:
            loss = None
        if modelConfig.training['optimizer']!=None:
            optimizer = modelConfig.training['optimizer']['class'](model.parameters(),**modelConfig.training['optimizer']['params'])
        else:
            optimizer = None
        trainer = modelConfig.training['training_class'](
            model, dataloader, validation_loader, 
            modelConfig.training['epochs'],
            loss, optimizer, config, outputManager
        )
        trainer.train()
    else:
        if modelConfig.modelParams['tokenizer']: #There must be an embedding layer -> extract dimension
            embedding_dim = torch.load(modelConfig.pretrained_model)['embedding.embedding.weight'].shape
            model = modelConfig.modelClass(tokenizer = embedding_dim, **modelConfig.modelParams['params'])
        model.load(modelConfig.pretrained_model)
    
    followingModelExists = modelIndex+1<len(config.models)
    followingModelDoInference = False
    if followingModelExists:
        followingModelDoInference = config.models[modelIndex+1].do_inference

    if modelConfig.do_inference or followingModelDoInference:
        if not modelConfig.from_previous:
            inferenceset = modelConfig.inference['dataset']['class'](**modelConfig.inference['dataset']['params'],
            validation_set=True, pre_trained_tokenizer = dataset.tokenizer)  
          
        inference_loader = modelConfig.inference['dataloader']['class'](inferenceset,**modelConfig.inference['dataloader']['params'])
        docs_summary = modelConfig.inference['method'](model,inference_loader,config.device,**modelConfig.inference['params'])

        if not followingModelExists:
            outputManager.saveSummaries(docs_summary.docs)