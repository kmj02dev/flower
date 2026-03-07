# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower message-based Decentralized FedAvg (DFedAvg) strategy."""

from collections.abc import Callable, Iterable
from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    ArrayRecord,
    MultiArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.server import Grid

from .dstrategy import DStrategy
from .strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    sample_nodes,
    validate_message_reply_consistency,
)


class DFedAvg(DStrategy):
    """Decentralized Federated Averaging strategy.

    Parameters
    ----------
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training.
    fraction_evaluate : float (default: 1.0)
        Fraction of nodes used during validation.
    min_train_nodes : int (default: 2)
        Minimum number of nodes used during training.
    min_evaluate_nodes : int (default: 2)
        Minimum number of nodes used during validation.
    min_available_nodes : int (default: 2)
        Minimum number of total nodes in the system.
    weighted_by_key : str (default: "num-examples")
        The key within each MetricRecord whose value is used as the weight.
    train_metrics_aggr_fn : Optional[callable] (default: None)
        Function used to aggregate MetricRecords from training round replies.
    evaluate_metrics_aggr_fn : Optional[callable] (default: None)
        Function used to aggregate MetricRecords from evaluation round replies.
    """

    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        peersrecord_key: str = "peers",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
    ) -> None:
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.weighted_by_key = weighted_by_key
        self.weighted_by_key = weighted_by_key
        self.arrayrecord_key = arrayrecord_key
        self.peersrecord_key = peersrecord_key
        self.configrecord_key = configrecord_key

        self.train_metrics_aggr_fn = train_metrics_aggr_fn or aggregate_metricrecords
        self.evaluate_metrics_aggr_fn = (
            evaluate_metrics_aggr_fn or aggregate_metricrecords
        )

        if self.fraction_evaluate == 0.0:
            self.min_evaluate_nodes = 0
            log(WARNING, "fraction_evaluate is set to 0.0.")
        if self.fraction_train == 0.0:
            self.min_train_nodes = 0
            log(WARNING, "fraction_train is set to 0.0.")

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "	├──> DFedAvg Sampling:")
        log(INFO, "	│	├──Fraction: train (%.2f) | evaluate ( %.2f)", self.fraction_train, self.fraction_evaluate)
        log(INFO, "	│	├──Minimum nodes: train (%d) | evaluate (%d)", self.min_train_nodes, self.min_evaluate_nodes)
        log(INFO, "	│	└──Minimum available nodes: %d", self.min_available_nodes)
        log(INFO, "	└──> Keys in records:")
        log(INFO, "		├── Weighted by: '%s'", self.weighted_by_key)

    def get_topology(self, grid: Grid) -> Dict[int, List[int]]:
        """Get the topology of the decentralized network. Defaulting to a ring topology."""
        node_ids = list(grid.get_node_ids())
        topology = {node_id: [] for node_id in node_ids}
        for i, node_id in enumerate(node_ids):
            topology[node_id] = [node_ids[(i+1)%len(node_ids)]]
        return topology

    def _sample_nodes(self, grid: Grid, fraction: float, min_nodes: int) -> List[int]:
        if fraction == 0.0:
            return []
        num_nodes = int(len(list(grid.get_node_ids())) * fraction)
        sample_size = max(num_nodes, min_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        return node_ids

    def configure_train(
        self, server_round: int, node_states: Dict[int, ArrayRecord], topology: Dict[int, List[int]], config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure nodes for local training. Send each their own model and peers' models."""
        node_ids = self._sample_nodes(grid, self.fraction_train, self.min_train_nodes)
        config["server-round"] = server_round

        messages = []
        for node_id in node_ids:
            if node_id not in node_states:
                continue
                
            record = RecordDict({
                "arrays": node_states[node_id], 
                "config": config
            })
            
            msg = Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id)
            messages.append(msg)
            
        return messages

    def configure_aggregate(
        self, server_round: int, node_states: Dict[int, ArrayRecord], topology: Dict[int, List[int]], config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Distribute neighbors' models for local fusion."""
        if not topology:
            log(WARNING, "No topology specified for configure_aggregate.")
            return []

        config["server-round"] = server_round
        
        messages = []
        # Send a message to each node that has neighbors
        for node_id, neighbors in topology.items():
            # Add neighbors' arrays natively via string IDs
            peers = MultiArrayRecord()
            for n_id in neighbors:
                if n_id in node_states:
                    peers[str(n_id)] = node_states[n_id]
                    
            # Node's own model needed for proper aggregation at Client
            if node_id in node_states:
                arrays = node_states[node_id]
            
            record = RecordDict(
                {
                    self.arrayrecord_key: arrays, 
                    self.peersrecord_key: peers, 
                    self.configrecord_key: config
                }
            )

            if len(record) > 1: # implies at least 1 neighbor model is attached
                msg = Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id)
                messages.append(msg)
                
        return messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[Dict[int, ArrayRecord] | None, Dict[int, MetricRecord] | None]:
        """Aggregate (extract) the newly trained models and metrics."""
        if not replies:
            return None, None
            
        arrays_dict: Dict[int, ArrayRecord] = {}
        metrics_dict: Dict[int, MetricRecord] = {}

        for msg in replies:
            if msg.has_error():
                continue
            src = msg.metadata.src_node_id
            
            # Extract models and metrics from normal TRAIN reply
            if "arrays" in msg.content:
                arrays_dict[src] = msg.content["arrays"]
            
            # For metrics, we assume it's stored under some key.
            # Usually ClientApp fit() returns metrics which Strategy receives.
            # Here we might need to extract the MetricRecord. Assuming "metrics" key is used.
            # standard RecordDict used by Flower might just put them in the generic content.
            # Flower >= 1.9 puts the MetricRecord under various keys, let's assume "metrics" if present.
            # If not, try to construct one out of scalar dictionary or return None.
            # DStrategy doesn't mandate the exact key, we just extract what we can.
            if "metrics" in msg.content and isinstance(msg.content["metrics"], MetricRecord):
                metrics_dict[src] = msg.content["metrics"]
            elif "fit_res" in msg.content: # Compatibility fallback
                pass

        return arrays_dict, metrics_dict

    def configure_evaluate(
        self, server_round: int, node_states: Dict[int, ArrayRecord], config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure decentral evaluation: Tell nodes to evaluate their current models."""
        node_ids = self._sample_nodes(grid, self.fraction_evaluate, self.min_evaluate_nodes)
        config["server-round"] = server_round

        messages = []
        for node_id in node_ids:
            if node_id not in node_states:
                continue
            
            record = RecordDict({
                "arrays": node_states[node_id], 
                "config": config
            })
            msg = Message(content=record, message_type=MessageType.EVALUATE, dst_node_id=node_id)
            messages.append(msg)
            
        return messages

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        """Aggregate MetricRecords into a single global unified MetricRecord (like Centralized FedAvg)."""
        if not replies:
            return None

        reply_contents = []
        for msg in replies:
            if not msg.has_error():
                reply_contents.append(msg.content)
                
        if not reply_contents:
            return None

        # Aggregate using the configured function
        metrics = self.evaluate_metrics_aggr_fn(
            reply_contents,
            self.weighted_by_key,
        )
        return metrics
