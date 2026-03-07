"""dfedavg: A Flower Baseline."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from dfedavg.dataset import load_data
from dfedavg.model import Net
from dfedavg.model import test as test_fn
from dfedavg.model import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # msg.content "config" no longer contains "step". 
    # Use peer model presence to determine train vs aggregate phase.
    peer_models_records = msg.content.get("peers", {})
            
    is_aggregation_phase = len(peer_models_records) > 0

    if not is_aggregation_phase:
        # Load the data
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        trainloader, _ = load_data(partition_id, num_partitions)
        local_epochs = context.run_config["local-epochs"]

        # Call the training function
        train_loss = train_fn(
            model,
            trainloader,
            local_epochs,
            device,
        )

        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }
    
    else:
        # Aggregation phase
        # 단순 가중 평균(FedAvg)을 통해 파라미터 업데이트
        # 자기 자신 모델 1개 + 이웃 모델 N개 -> 총 N+1개의 모델을 동일한 가중치로 평균
        if peer_models_records:
            model_state = model.state_dict()
            num_models = len(peer_models_records) + 1
            
            # Initialize peer dict states
            peer_states = [peer_record.to_torch_state_dict() for peer_record in peer_models_records.values()]
            
            for key in model_state.keys():
                # Sum 자기 자신 + Peer models
                total_tensor = model_state[key].clone()
                for peer_state in peer_states:
                    total_tensor += peer_state[key]
                
                # 평균 계산
                model_state[key] = total_tensor / num_models
                
            # 병합된 가중치로 모델 갱신
            model.load_state_dict(model_state)

        metrics = {}

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    _, valloader = load_data(partition_id, num_partitions)

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
