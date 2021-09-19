#!/bin/bash

wget https://storage.googleapis.com/recipe-box/recipes_raw.zip
mkdir recipes-dataset
unzip recipes_raw.zip -d recipes-dataset
rm recipes_raw.zip
