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
"""Unit tests for MultiArrayRecord."""

import json
import sys
import unittest
from collections import OrderedDict
from types import ModuleType
from typing import Any
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from parameterized import parameterized

from flwr.common import ndarray_to_bytes

from ..constant import SType
from ..inflatable import get_object_body, get_object_type_from_object_content
from ..typing import NDArray
from .array import Array
from .arrayrecord import ArrayRecord
from .multiarrayrecord import MultiArrayRecord


def _get_buffer_from_ndarray(array: NDArray) -> bytes:
    """Return a bytes buffer from a given NumPy array."""
    from io import BytesIO
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


class TorchTensor(Mock):
    """Mock Torch tensor class."""


MOCK_TORCH_TENSOR = TorchTensor(numpy=lambda: np.array([[1, 2, 3]]))
MOCK_TORCH_TENSOR.detach.return_value = MOCK_TORCH_TENSOR
MOCK_TORCH_TENSOR.cpu.return_value = MOCK_TORCH_TENSOR


class TestMultiArrayRecord(unittest.TestCase):
    """Unit tests for MultiArrayRecord."""

    def setUp(self) -> None:
        """Set up the test case."""
        # Patch torch
        self.torch_mock = Mock(spec=ModuleType, Tensor=TorchTensor)
        self._original_torch = sys.modules.get("torch")
        sys.modules["torch"] = self.torch_mock

    def tearDown(self) -> None:
        """Tear down the test case."""
        # Unpatch torch
        del sys.modules["torch"]
        if self._original_torch is not None:
            sys.modules["torch"] = self._original_torch

    @parameterized.expand(  # type: ignore
        [
            (
                {"client_1": [np.array([1, 2])], "client_2": [np.array([3, 4])]},
            ),
            (
                {"client_1": [np.array(5)]},
            ),
            (
                {},
            ),
        ]
    )
    def test_from_numpy_ndarrays_dict(self, ndarrays_dict: dict[str, list[NDArray]]) -> None:
        """Test creating a MultiArrayRecord from a dictionary of NumPy arrays."""
        with patch.object(ArrayRecord, "from_numpy_ndarrays") as mock_from_numpy:
            # Prepare
            mock_records = [Mock(spec=ArrayRecord) for _ in ndarrays_dict]
            mock_from_numpy.side_effect = mock_records
            expected_keys = list(ndarrays_dict.keys())

            # Execute
            record = MultiArrayRecord.from_numpy_ndarrays_dict(ndarrays_dict.copy())

            # Assert
            self.assertEqual(list(record.keys()), expected_keys)
            self.assertEqual(list(record.values()), mock_records)
            mock_from_numpy.assert_has_calls(
                [call(arr_list, keep_input=True) for arr_list in ndarrays_dict.values()], 
                any_order=False
            )

    def test_from_torch_state_dicts_with_torch(self) -> None:
        """Test creating a MultiArrayRecord from PyTorch state_dicts."""
        # Prepare
        state_dicts = {
            "model_A": {"weight": TorchTensor(), "bias": TorchTensor()},
            "model_B": {"weight": TorchTensor()},
        }
        mock_records = [Mock(spec=ArrayRecord), Mock(spec=ArrayRecord)]
        
        with patch.object(ArrayRecord, "from_torch_state_dict") as mock_from_torch:
            mock_from_torch.side_effect = mock_records

            # Execute
            record = MultiArrayRecord.from_torch_state_dicts(state_dicts)

            # Assert
            self.assertEqual(list(record.keys()), list(state_dicts.keys()))
            mock_from_torch.assert_has_calls(
                [call(sd, keep_input=True) for sd in state_dicts.values()], any_order=False
            )
            self.assertEqual(list(record.values()), mock_records)

    def test_from_torch_state_dicts_without_torch(self) -> None:
        """Test `MultiArrayRecord.from_torch_state_dicts` without PyTorch."""
        with patch.dict("sys.modules", {}, clear=True):
            with self.assertRaises(RuntimeError) as cm:
                MultiArrayRecord.from_torch_state_dicts({})
            self.assertIn("PyTorch is required", str(cm.exception))

    def test_to_numpy_ndarrays_dict(self) -> None:
        """Test converting a MultiArrayRecord to a dictionary of NumPy arrays."""
        # Prepare
        record = MultiArrayRecord()
        mock_record_a = Mock(spec=ArrayRecord)
        mock_record_b = Mock(spec=ArrayRecord)
        
        expected_a = [np.array([1, 2])]
        expected_b = [np.array([3, 4])]
        
        mock_record_a.to_numpy_ndarrays.return_value = expected_a
        mock_record_b.to_numpy_ndarrays.return_value = expected_b
        
        record["model_a"] = mock_record_a
        record["model_b"] = mock_record_b

        # Execute
        result = record.to_numpy_ndarrays_dict()

        # Assert
        self.assertEqual(result["model_a"], expected_a)
        self.assertEqual(result["model_b"], expected_b)
        mock_record_a.to_numpy_ndarrays.assert_called_once()
        mock_record_b.to_numpy_ndarrays.assert_called_once()

    def test_init_no_args(self) -> None:
        """Test initializing with no arguments."""
        _ = MultiArrayRecord()

    @parameterized.expand([(True,), (False,)])  # type: ignore
    def test_init_record_dict_keep_input_false(self, use_keyword: bool) -> None:
        """Test initializing with an record_dict and keep_input=False."""
        # Prepare
        arr_rec = ArrayRecord({"x": Array(dtype="float32", shape=(2, 2), stype=SType.NUMPY, data=b"data")})
        rec_dict = {"client_1": arr_rec}

        # Execute
        if use_keyword:
            record = MultiArrayRecord(record_dict=rec_dict, keep_input=False)
        else:
            record = MultiArrayRecord(rec_dict, keep_input=False)

        # Assert
        self.assertEqual(record["client_1"], arr_rec)
        self.assertEqual(len(rec_dict), 0)

    @parameterized.expand(  # type: ignore
        [
            ((42,), {}),
            (("invalid",), {}),
            ((), {"numpy_ndarrays_dict": {"c1": [np.array([2])]}, "record_dict": {"c1": Mock(spec=ArrayRecord)}}),
        ]
    )
    def test_init_unrecognized_arg_raises_error(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Test initializing with unrecognized arguments."""
        with self.assertRaisesRegex(TypeError, "Invalid arguments for MultiArrayRecord.*"):
            MultiArrayRecord(*args, **kwargs)

    @parameterized.expand(  # type: ignore
        [
            ({"client_1": [np.array([1, 2])], "client_2": [np.array([3, 4])]},),
            ({"single_client": [np.array(5)]},),
            ({},),
        ]
    )
    def test_inflation_deflation(self, array_dict_content) -> None:
        """Test inflation and deflation of MultiArrayRecord."""
        # Prepare actual ArrayRecords instead of mocks for inflation tests
        record_content = {
            k: ArrayRecord(v) for k, v in array_dict_content.items()
        }
        multi_rec = MultiArrayRecord(record_content)

        # Assert Expected children
        assert multi_rec.children == {rec.object_id: rec for rec in multi_rec.values()}

        multi_rec_b = multi_rec.deflate()

        # Assert Class name matches
        assert (
            get_object_type_from_object_content(multi_rec_b)
            == multi_rec.__class__.__qualname__
        )
        
        # Body of deflated MultiArrayRecord matches its direct JSON serialization
        record_refs = {name: rec.object_id for name, rec in multi_rec.items()}
        record_refs_enc = json.dumps(record_refs).encode("utf-8")
        assert get_object_body(multi_rec_b, MultiArrayRecord) == record_refs_enc

        # Inflate
        if len(array_dict_content) > 0:
            with pytest.raises(ValueError):
                MultiArrayRecord.inflate(multi_rec_b)

        # Inflate passing children
        multi_rec_ = MultiArrayRecord.inflate(multi_rec_b, children=multi_rec.children)

        # Assert both objects are identical in IDs
        assert multi_rec.object_id == multi_rec_.object_id

    def test_count_bytes(self) -> None:
        """Test bytes in a MultiArrayRecord are computed correctly."""
        # Create a mock ArrayRecord with a fixed byte count
        mock_rec_1 = Mock(spec=ArrayRecord)
        mock_rec_1.count_bytes.return_value = 100
        
        mock_rec_2 = Mock(spec=ArrayRecord)
        mock_rec_2.count_bytes.return_value = 150

        key_1 = "client_A"
        key_2 = "client_B"
        
        multi_record = MultiArrayRecord({key_1: mock_rec_1, key_2: mock_rec_2})
        
        # The total bytes should be the sum of children bytes + length of the keys
        expected_bytes = 100 + 150 + len(key_1) + len(key_2)
        assert multi_record.count_bytes() == expected_bytes