

# Installation

- Download the repository
- Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
- Add your data in the data folder (must be in the [correct](Data-format) format!)
- [Configure](Configuration) the pipeline
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
- Output configuration

## Modules pipeline
Each element in the list *model* has the following format
- **name**: specify the name of the model class
- **config**: specify the path of the configuration file (starting from ./config)
- **train**: if true, compute train for the model on data specified in the config file of the module
- **pretrained_model**: path of the pretrained model
- **inference**: if true, compute summarization of documents specified in the config file of the module
- **from_previous**: specify if the module takes as input the output of the previous module or reads data directly from the specified files. (true for the first module raise an exception)

## Module config file
Each module must have a proper configuration file.

