# How to produce summaries

- Download the repository
- Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
- Add your data in the data folder (must be in the [correct](#Data-format) format!)
- [Configure](#Configuration) the pipeline (the default is ContentRanking + Bart)
- Launch [train.py](./train.py)
    ```bash
    python train.py
    ```

# Data format
Data must be provided in JSON format with the following structure:
```bash
{
    article_id: str
    abstract_text: List[str]
    article_text: List[str]
    section_names: List[str]
    sections: List[List[str]]
}
```

# Configuration
The main configuration file is [config.json](config/config.json)
In this file are defined:
- Device on which execute the summarizer
- Modules pipeline
- Output configuration:
    - name of the folder in which the log and the models will be saved
    - flags for saving models and log file
    - wandb: if null, wandb is not invoked. To set up wandb, set this field to:
        ```bash
        {
            "project":"Name of the project",
            "entity":"Account"
        }
        ```

## Modules pipeline
Each element in the list *model* has the following format
- **name**: specify the name of the model class
- **config**: specify the path of the configuration file (starting from ./config)
- **train**: if true, compute train for the model on data specified in the config file of the module
- **pretrained_model**: path of the pretrained model
- **inference**: if true, compute summarization of documents specified in the config file of the module
- **from_previous**: specify if the module takes as input the output of the previous module or reads data directly from the specified files. (true for the first module raise an exception)

## Module config file
Each module must have a proper configuration file.\
Each configuration file must have:
- **params**: all the parameters of the model
- **tokenizer**: the tokenizer class and all the tokenizer params
- **train**: the trainer class
    - **training_dataset**: the dataset class, the dataset params and the train data path
    - **validation_dataset**: the validation dataset class,its params and the validation data path
    - **training_dataloader**: the training dataloader and its params
    - **training_epochs**: for how many epochs compute the training
    - **optimizer**: the optimizer class and its params
    - **loss**: the loss class and its params

# How to add a module
- Add a model class in [models](src/models) folder
- Add a dataset class in [datasets](src/datasets) folder
- Add a dataloader class in [dataloaders](src/dataloaders) folder
- Add a tokenizer class in [tokenizers](src/tokenizers) folder
- Add a training class in [trainings](src/trainings) folder
- If needed, add a loss class in [losses](src/losses) folder
- Add the proper static method in [inference.py](src/utils/inference.py)
- If needed, add the proper static method in [wrapper.py](src/utils/wrapper.py)

## How a module works
Each module, with its own dataset, can read data from file (in [correct](#Data-format) format) or from the output of the previous module in the pipeline. The latter option is made possible by the wrapper class, that converts the [standard](src/datasets/standardDataset.py) dataset format to the one specific for this module.\
Each module, in inference time, produces its own output in a [standard](src/datasets/standardDataset.py) dataset format.