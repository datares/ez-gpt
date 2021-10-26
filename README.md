# Recipe GPT - Using GPT2 to generate recipes

## Setup

First clone the repository: 
```
git clone https://github.com/datares/recipe-gpt.git
cd recipe-gpt
```

To setup the development environment, create a new conda environment with the following command
```
conda create -n gpt python=3.8
pip install -r requirements.txt
```

Then to download the dataset, run
```
./scripts/data.sh
```
which will download json files containing many recipes into the `data` directory.  There are are around 
124000 recipes.

## Running the Model
The model was tested using pytorch version 1.9.0 and cuda version 11.2.

To start training, run
```
python main.py
```
Model checkpoints will be saved automatically to the `checkpoints` directory, so training can be restarted easily by
reloading one of the checkpoints.

A generated recipe is below

### surroundRed Roasted Roasted Chicken 
1. 1 (4 ounce) package chicken broth  
2. 2 (6.5 ounce) cans whole grain tomatoes, thawed  
3. 1 (8 ounce) can diced black olives  
4. 1/2 cup shredded fresh cilantro, or to taste  salt and pepper to taste

Preheat oven to 375 degrees F (190 degrees C).
Ladle chicken, tomatoes, olives, lettuce, lettuce, cilantro, and salt into a large skillet. Cook well in the preheated oven until tomatoes are tender, about 1/8 of the time, and are tender. Remove from oven and let cool slightly. Remove chicken from pot; drain excess water from the cooking grate.

disclaimer: quality of recipes cannot be guarenteed :)
