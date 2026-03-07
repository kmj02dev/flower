"""dfedavg: A Flower Baseline."""

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import DFedAvg

import flwr.serverapp.strategy.dstrategy
from flwr.serverapp.strategy.result import HeterogeneousResult
flwr.serverapp.strategy.dstrategy.HeterogeneousResult = HeterogeneousResult

from dfedavg.model import Net

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]

    # Load global model
    global_model = Net()
    # Provide an initial array for each node in the grid
    initial_arrays = [ArrayRecord(global_model.state_dict()) for _ in grid.get_node_ids()]

    # Initialize DFedAvg strategy
    strategy = DFedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=1,
    )

    # Start strategy, run DFedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    for node_id, node_state in result.node_states.items():
        state_dict = node_state.to_torch_state_dict()
        
        filename = f"final_model_node_{node_id}.pt"
        torch.save(state_dict, filename)
        print(f"Successfully saved model for node {node_id} as {filename}")
