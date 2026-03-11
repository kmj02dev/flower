"""$project_name: A Flower / $framework_str app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from $import_name.dataset import load_data
from $import_name.model import create_model
from $import_name.model import test as test_fn
from $import_name.model import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = create_model(**context.run_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device(f"cuda:{context.run_config['cuda_id']}" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, _ = load_data(partition_id, num_partitions, **context.run_config)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        device,
        **context.run_config
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = create_model(**context.run_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device(f"cuda:{context.run_config['cuda_id']}" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    _, valloader = load_data(partition_id, num_partitions, **context.run_config)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
