# LID Task using BoW approach.

## Download the data

Use the following command to download the dataset that I used for this task.

Create a `/data` in the root folder of this project and run the following inside `/data`.

```
wget https://downloads.tatoeba.org/exports/sentences.csv
```

## Prepare the dataset

```python
python dataset.py --prepare_data --save
```

## Run the training task

```python
python trainer.py
```

You can check the trainer.py file to check for the hyperparameters that you wish to change.
