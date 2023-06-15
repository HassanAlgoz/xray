# Classification using Transformers in PyTorch

This repo is for learning purposes.

| Author | Hassan Algoz | 
|:-|:-|
| Acknowledgement | this repo is adapted from: https://cs230.stanford.edu/blog/pytorch/ |

My work on the repo was:

- add code piece by piece to make sure I get to understand it
- build the transformer following the ["Attention Is All You Need" paper](https://arxiv.org/abs/1706.03762) (decoder is yet to be implemented)
   - However, the encoder was enough to build a sentiment analysis model
- using DirectML for Radeon GPU support
- add `explore.ipynb` for understanding the data
- add `inspect.ipynb` for trained model analysis
- simplify data loader
- use torch 2.0 (thanks to backward compatability with torch 1.x, I did not face any issues with this)

## Task

Given a text, can the model predict its label? e.g., spam/ham or positive/negative

## The Dataset

1. Visit https://www.kaggle.com/code/matleonard/text-classification/input?select=yelp_ratings.csv and look for `yelp_ratings.csv` and download it.

2. Place the `yelp_ratings.csv` as `data/yelp_ratings.csv`

3. Explore the data using `explore.ipynb`

4. Run the notebook `split_data.ipynb` to split it into: `trian, test, dev` sets

Here is what `yelp_ratings.csv` looks like:

```
"sentiment","text"
1,"Excellent food and staff.
 I hope the course hasn't undergone any changes like the restaurant atmosphere and food has!"
0,"I really, really wanted to like The Chickery. I imagine this is what prison food is like."
1,"Sabrina is my stylist. She always has great advice about how to style my hair and what products to use."
```

- Note that we have dropped the `stars` column, and have made the `sentiment` the first column.
- Small data can be useful in searching for hyper-parameters. Then, for actual training, the more the better.


## Explore

The purpose of `explore.ipynb` is for gaining a better understanding of the data. Things like:

- Check for missing values
- Look at the distribution of the target variable
- Investigate relationships between features
- Visualize the data in various ways
- Check for any outliers or anomalies
- Gain insights that will help inform further analysis and modeling


## Experimentation

We created a `base_model` directory for you under the `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the experiment. It looks like

```json
{
  "learning_rate": 1e-3,
  "batch_size": 5,
  "num_epochs": 2
}
```

For every new experiment, you will need to create a new directory under `experiments` with a `params.json` file.

To **Train** your experiment. Simply run:

```sh
python train.py --data_dir data/small --model_dir experiments/base_model
```

It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the development set.


## Hyperparameters search

We created a new directory `learning_rate` in `experiments` for you. Now, run:

```sh
python search_hyperparams.py --data_dir data/small --parent_dir experiments/learning_rate
```

It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`.

**Display the results** of the hyperparameters search in a nice format:

```sh
python synthesize_results.py --parent_dir experiments/learning_rate
```


## Evaluation on the test set

Once you've run many experiments and selected your best model and hyperparameters based on the performance on the development set, you can finally evaluate the performance of your model on the test set. Run

```sh
python evaluate.py --data_dir data/small --model_dir experiments/base_model
```

Note that `evaluate.py` has an `evaluate()` function, which `train.py` use to evaluate against the `val`idation set, whereas doing `python evaluate.py` does the evaluation against the `test` set.


## Inspect

The purpose of `inspect.ipynb` is to analyze the model.

- Identify any potential issues
   - Consistently incorrect labels for certain examples
   - Overfitting or underfitting certain types of examples
- Investigate how the model made predictions by:
   - Reviewing the most important features
   - Examining how changes to the input values affect the predictions
- Form hypotheses for how to improve the model


## File structure semantics

- `data`
   - `embeddings`         - input vocabulary as vectors
   - `reader.py`          - defines how to read the dataset
- `evaluate.py`
- `experiments`
   - `base_model`
      - `params.json`     - hyperparameters for this experiment
- `model`
   - `attention_head.py`
   - `encoder.py`
   - `layer_norm.py`
   - `net.py`         - neural network, loss function and metrics
- `requirements.txt`
- `split_data.ipynb`  - the notebook that splits the data
- `train.ipynb`       - same as train.py but used in blocks
- `train.py`          - defines how the model is trained
- `utils.py`          - general helpful functions
