"""Evaluates the model"""

import argparse
import logging
import os
from typing import Callable, Dict, Iterator

import numpy as np
import torch
import torch_directml
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

import utils
from utils import Params
import model.net as net
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
    default="best",
    help="name of the file in --model_dir \
                     containing weights to load",
)


def evaluate(
    *,
    device,
    model: torch.nn.Module,
    loss_fn: Callable,
    data_iterator: Iterator,
    metrics: Dict[str, Callable],
    params: Params,
    num_steps: int
) -> Dict[str, float]:
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for _ in range(num_steps):
        data = next(data_iterator, None)
        if data == None:
            break
        input = data[0].to(device)
        expected_output = data[1].to(device)

        # compute model output
        output = model(input)
        loss = loss_fn(output, expected_output)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output = output.data.cpu().numpy()
        expected_output = expected_output.data.cpu().numpy()

        # compute all metrics on this batch
        summary = {
            metric: metrics[metric](output, expected_output) for metric in metrics
        }
        summary["loss"] = loss.item()
        summ.append(summary)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in metrics}
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items()
    )
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == "__main__":
    """
    Evaluate the model on the test set.
    """
    # fixed seed for reproducible experiments
    seed = 230

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(args.model_dir, "evaluate.log"))

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
    logging.info("Creating the dataset...")

    # load data
    test_dataset = Dataset(
        "test", args.data_dir, embeddings_kv, params.max_input_length
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=params.batch_size, shuffle=True
    )
    params.test_size = len(test_dataset)

    # Define the model
    logging.info("Initializing model...")
    model = net.Net(
        device=device,
        embeddings=embeddings,
        num_heads=params.num_heads,
        num_layers=params.num_layers,
        num_classes=params.num_classes,
        input_window_size=params.max_input_length,
    ).to(device)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(
        os.path.join(args.model_dir, args.restore_file + ".pth.tar"), model
    )

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    test_metrics = evaluate(
        device=device,
        model=model,
        loss_fn=loss_fn,
        data_iterator=iter(test_data_loader),
        metrics=metrics,
        params=params,
        num_steps=num_steps,
    )
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file)
    )
    utils.save_dict_to_json(test_metrics, save_path)
