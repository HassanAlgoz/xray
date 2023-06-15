"""Train the model"""

from typing import Callable, Dict, Iterator
import argparse
import logging
import os

import torch
import torch.optim as optim
import torch_directml
from gensim.models import KeyedVectors
import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader

import utils
from utils import Params
import model.net as net
from evaluate import evaluate
from data.reader import Dataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", default="data/small", help="Directory containing the dataset"
)
parser.add_argument(
    "--model_dir",
    default="experiments/base_model",
    help="Directory containing params.json",
)
parser.add_argument(
    "--restore_file",
    default=None,
    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training",
)  # 'best' or 'train'


def train(
    *,
    device,
    model: torch.nn.Module,
    optimizer: torch.optim,
    loss_fn: Callable,
    data_iterator: Iterator,
    metrics: Dict[str, Callable],
    num_steps: int,
    params: Params,
):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        num_steps: (int) number of batches to train on
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        data = next(data_iterator, None)
        if data == None:
            break
        input = data[0].to(device)
        expected_output = data[1].to(device)

        # compute model output and loss
        output = model(input)
        loss = loss_fn(output, expected_output)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output = output.data.cpu().numpy()
            expected_output = expected_output.data.cpu().numpy()

            # compute all metrics on this batch
            summary = {
                metric: metrics[metric](output, expected_output) for metric in metrics
            }
            summary["loss"] = loss.item()
            summ.append(summary)

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss="{:05.3f}".format(loss_avg()))

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in metrics}
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items()
    )
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(
    *,
    device,
    model: torch.nn.Module,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    optimizer: torch.optim,
    loss_fn: Callable,
    metrics: Dict[str, Callable],
    params: Params,
    model_dir: str,
    restore_file: str = None,
) -> None:
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validaion data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + ".pth.tar")
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        logging.info(f"Epoch {epoch+1}/{params.num_epochs}")

        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = (params.train_size + 1) // params.batch_size
        train(
            device=device,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            data_iterator=iter(train_data_loader),
            metrics=metrics,
            params=params,
            num_steps=num_steps,
        )

        # Evaluate for one epoch on validation set
        num_steps = (params.val_size + 1) // params.batch_size
        val_metrics = evaluate(
            device=device,
            model=model,
            loss_fn=loss_fn,
            data_iterator=iter(val_data_loader),
            metrics=metrics,
            params=params,
            num_steps=num_steps,
        )

        val_acc = val_metrics["accuracy"]
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
            },
            is_best=is_best,
            checkpoint=model_dir,
        )

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == "__main__":
    # fixed seed for reproducible experiments
    seed = 230

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(args.model_dir, "train.log"))

    # select the GPU device if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed)
    elif torch_directml.is_available():
        device = torch_directml.device(torch_directml.default_device())
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)

    logging.info("Loading embeddings...")
    embeddings_kv = KeyedVectors.load_word2vec_format(
        "./data/embeddings/glove.6B.100d.txt", binary=False, no_header=True
    )
    embeddings = torch.tensor(embeddings_kv.vectors)

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    train_dataset = Dataset(
        "train", args.data_dir, embeddings_kv, params.max_input_length
    )
    train_data_loader = DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True
    )
    params.train_size = len(train_dataset)

    val_dataset = Dataset("val", args.data_dir, embeddings_kv, params.max_input_length)
    val_data_loader = DataLoader(
        val_dataset, batch_size=params.batch_size, shuffle=True
    )
    params.val_size = len(val_dataset)

    # Define the model and optimizer
    logging.info("Initializing model...")
    # model = net.Net(params).cuda() if params.cuda else net.Net(params)
    model = net.Net(
        device=device,
        embeddings=embeddings,
        num_heads=params.num_heads,
        num_layers=params.num_layers,
        num_classes=params.num_classes,
        input_window_size=params.max_input_length,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Start training for {} epoch(s)...".format(params.num_epochs))
    train_and_evaluate(
        device=device,
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        params=params,
        model_dir=args.model_dir,
        restore_file=args.restore_file,
    )
