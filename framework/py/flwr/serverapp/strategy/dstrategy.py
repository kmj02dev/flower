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
"""Flower message-based strategy."""


import io
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from logging import INFO
from typing import Dict, List

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, log
from flwr.server import Grid

from .result import Result
from .strategy_utils import log_strategy_start_info


class DStrategy(ABC):
    """Abstract base class for decentralized strategy implementations."""

    @abstractmethod
    def get_topology(self, grid: Grid) -> Dict[int, List[int]]:
        """Get the topology of the decentralized network.

        Parameters
        ----------
        grid : Grid
            The Grid instance used for node sampling and communication.

        Returns
        -------
        Dict[int, List[int]]
            The topology of the decentralized network.
        """
        
    @abstractmethod
    def configure_train(
        self, server_round: int, node_states: Dict[int, ArrayRecord], topology: Dict[int, List[int]], config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        node_states : Dict[int, ArrayRecord]
            The arrays of each node.
        topology : Dict[int, List[int]]
            The topology of the decentralized network.
        config : ConfigRecord
            Configuration to be sent to clients nodes for training.
        grid : Grid
            The Grid instance used for node sampling and communication.

        Returns
        -------
        Iterable[Message]
            An iterable of messages to be sent to selected client nodes for training.
        """

    @abstractmethod
    def configure_aggregate(
        self, server_round: int, node_states: Dict[int, ArrayRecord], topology: Dict[int, List[int]], config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        node_states : Dict[int, ArrayRecord]
            The arrays of each node.
        topology : Dict[int, List[int]]
            The topology of the decentralized network.
        config : ConfigRecord
            Configuration to be sent to clients nodes for training.
        grid : Grid
            The Grid instance used for node sampling and communication.

        Returns
        -------
        Iterable[Message]
            An iterable of messages to be sent to selected client nodes for training.
        """

    @abstractmethod
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Dict[int, ArrayRecord] | None, Dict[int, MetricRecord] | None]:
        """Aggregate training results from client nodes.

        Parameters
        ----------
        server_round : int
            The current round of federated learning, starting from 1.
        replies : Iterable[Message]
            Iterable of reply messages received from client nodes after training.
            Each message contains ArrayRecords and MetricRecords that get aggregated.

        Returns
        -------
        tuple[Optional[Dict[int, ArrayRecord]], Optional[Dict[int, MetricRecord]]]
            A tuple containing:
            - Dict[int, ArrayRecord]: Aggregated ArrayRecord, or None if aggregation failed
            - Dict[int, MetricRecord]: Aggregated MetricRecord, or None if aggregation failed
        """

    @abstractmethod
    def configure_evaluate(
        self, server_round: int, nodes_arrays: Dict[int, ArrayRecord], config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of evaluation.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        arrays : ArrayRecord
            Current global ArrayRecord (e.g. global model) to be sent to client
            nodes for evaluation.
        config : ConfigRecord
            Configuration to be sent to clients nodes for evaluation.
        grid : Grid
            The Grid instance used for node sampling and communication.

        Returns
        -------
        Iterable[Message]
            An iterable of messages to be sent to selected client nodes for evaluation.
        """

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate evaluation metrics from client nodes.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        replies : Iterable[Message]
            Iterable of reply messages received from client nodes after evaluation.
            MetricRecords in the messages are aggregated.

        Returns
        -------
        Optional[MetricRecord]
            Aggregated evaluation metrics from all participating clients,
            or None if aggregation failed.
        """

    @abstractmethod
    def summary(self) -> None:
        """Log summary configuration of the strategy."""

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def start(
        self,
        grid: Grid,
        initial_arrays: List[ArrayRecord],
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, List[ArrayRecord]], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the decentralized federated learning strategy.

        Runs the complete decentralized federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, List[ArrayRecord]], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        log(INFO, "Starting %s strategy:", self.__class__.__name__)

        log_strategy_start_info(
            num_rounds, initial_arrays[0], train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = HeterogeneousResult()
        
        # Initialize node states
        if len(initial_arrays) != len(grid.get_node_ids()):
            raise ValueError("Number of initial arrays must match number of nodes")
        
        node_states = {}
        for i, node_id in enumerate(grid.get_node_ids()):
            node_states[node_id] = initial_arrays[i]
                
        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, node_states)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            topology = self.get_topology(grid)
            
            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    node_states,
                    topology,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            agg_node_states, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Aggregate train
            agg_replies = grid.send_and_receive(
                messages=self.configure_aggregate(
                    current_round,
                    node_states,
                    topology,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            agg_node_states, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_node_states is not None:
                result.node_states = agg_node_states
                for node_id, array_record in agg_node_states.items():
                    node_states[node_id] = array_record
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    node_states,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, node_states)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
