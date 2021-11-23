# EZ-GPT: Dead simple language modeling

The goal of this project is to use GPT-2 to finetune on downstream language modeling tasks, and in the process 
prepare a repository that makes it as simple as possible to train a text generation model on any dataset.

## Setup

First clone the repository: 
```
git clone https://github.com/datares/ez-gpt.git
cd ez-gpt
```

To setup the development environment, create a new conda environment with the following command
```
conda create -n gpt python=3.8
conda activate gpt
pip install -r requirements.txt
```

The repository comes setup with four separate language modeling tasks
- recipes
- Shakespeare plays
- Drake lyrics
- stack overflow posts

Then to download the dataset, run
```
./scripts/data.sh
```
which will download data files into the `data` directory.

## Configuration
There is a `config.json` file to hangle the training configuration and hyperparameters
```json
{
    "dataset": "data/shakespeare.json",
    "dataset_type": "shakespeare",
    "load_from_checkpoint": false,
    "checkpoint_path": "",
    "learning_rate": 5e-4,
    "batch_size": 1,
    "epsilon": 1e-8,
    "sample_every": 100,
    "max_epochs": 25,
    "opt_warmup_steps": 1e2,
    "precision": 16,
    "fast_train": false,
    "logging_level": "INFO"
}
```

## Running the Model
The model was tested using pytorch version 1.9.0 and cuda version 11.2.

To start training on the config, run
```
python main.py
```
Model checkpoints will be saved automatically to the `checkpoints` directory, so training can be restarted easily by
reloading one of the checkpoints.

<!-- A generated recipe is below

### surroundRed Roasted Roasted Chicken 
1. 1 (4 ounce) package chicken broth  
2. 2 (6.5 ounce) cans whole grain tomatoes, thawed  
3. 1 (8 ounce) can diced black olives  
4. 1/2 cup shredded fresh cilantro, or to taste  salt and pepper to taste

Preheat oven to 375 degrees F (190 degrees C).
Ladle chicken, tomatoes, olives, lettuce, lettuce, cilantro, and salt into a large skillet. Cook well in the preheated oven until tomatoes are tender, about 1/8 of the time, and are tender. Remove from oven and let cool slightly. Remove chicken from pot; drain excess water from the cooking grate.

disclaimer: quality of recipes cannot be guarenteed :) -->

## Generating Text
To generate text using a trained model, use one of the checkpoints that were saved during training.  

The `generate.py` script handles all text generation for the model.  The script takes one argument, namely
the path of the checkpoint to use.  

A sample usage is shown below
```bash
python generate.py checkpoints/shakespeare/shakespeare.ckpt
```

Sample outputs for the Shakespeare generation are shown below

```txt
CARDINAL WOLSEY
        [Within]    Pray you, I pray you, have some notice of my head.
HELENA
        No sooner love but the love, that should be so. But when I was not the wife is so to my marriage.
MISTRESS FORD
        You have my letter to his master, I will tell your mind.
LORENZO
        My lord, to her, my lord, I thank her, and, sir. I have heard much like to her at the time, 
        but that I am content to do any thing, I know not what: She brings all that do. 
        I must have her on her.
FLUELLEN
        Why, she's very strange, indeed, she has a great cause, Wh she hath a head to me: for that's 
        a true man. Exit Re-enter Bawd Boult
KING HENRY IV
        A strange thought shall be known to some other, And he will come to the court. It cannot 
        be but a thousand, Nor never did him till that all the rest. Now he'll speak, and never will he 
        live, Nor must I not live, Till 'tis but a man, but never, For there's a man.
LORD POLONIUS
        'Tis true: and I am sorry To what I'll say.
BENEDICK
        I knew it not: if it please me so, I could never be out, for here.
First Servingman
        I have thought the best words to tell your lordship.
KING HENRY VI 
        O, your highness knows the matter I'll tell me all the rest: so you may to me know: if you 
        do meet them not in a way meet them to-morrow, your highness comes a very hand: I will 
        speak with them at this last, and to them, let them be a subject to the field: and when 
        I meet him, they'll draw upon us, make an end of him.
```

## Extending
It is very simple to use this repository to train a language model on a new dataset.

Only a few changes need to be made
1. Add a PyTorch dataset class to `datasets/train/` to define how data should be loaded into the model
2. Import your PyTorch dataset in  `datasets/DataModule.py` and add the name of your dataset to the `ds_table` variable.  The key is the name of the dataset as it pertains to the `config.json`
3. Update the `config.json`
   1. `dataset` should be the path to the data to train the model
   2. `dataset_type` should be the name of the dataset as defined in `DataModule.py`
4. Start training by running `python main.py`

The training losses can be viewed on Weights and Biases and the lowest three validation loss checkpoints are stored in the `checkpoints` directory.

Hopefully this illustrates how simple it can be to train a language model on a downstream task with this repository.  Only 2 changes are needed in the actual codebase.
