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
"""MultiArrayRecord."""

from __future__ import annotations

import gc
import json
import sys
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from ..constant import GC_THRESHOLD
from ..inflatable import InflatableObject, add_header_to_object_body, get_object_body
from ..logger import log
from ..typing import NDArray
from .array import Array
from .typeddict import TypedDict
from .arrayrecord import ArrayRecord

if TYPE_CHECKING:
    import torch


def _raise_multi_array_record_init_error() -> None:
    raise TypeError(
        f"Invalid arguments for {MultiArrayRecord.__qualname__}. Expected either "
        "a dictionary of ArrayRecords, a dictionary of lists of NumPy ndarrays, "
        "or a dictionary of PyTorch state_dicts. "
        "The `keep_input` argument is keyword-only."
    )


def _check_key(key: str) -> None:
    """Check if key is of expected type."""
    if not isinstance(key, str):
        raise TypeError(f"Key must be of type `str` but `{type(key)}` was passed.")


def _check_value(value: ArrayRecord) -> None:
    """Check if value is of expected type (ArrayRecord)."""
    if not isinstance(value, ArrayRecord):
        raise TypeError(
            f"Value must be of type `{ArrayRecord}` but `{type(value)}` was passed."
        )


class MultiArrayRecord(TypedDict[str, ArrayRecord], InflatableObject):
    """Multi Array record.

    A typed dictionary (``str`` to :class:`ArrayRecord`) that can store multiple 
    models or named array collections. Useful for storing states from multiple 
    clients, ensemble models, or personalized model architectures.

    Internally, this behaves similarly to a ``dict[str, ArrayRecord]``.
    This object is fully serializable and acts as a container of InflatableObjects.

    Examples
    --------
    Initializing an empty MultiArrayRecord::

        record = MultiArrayRecord()

    Initializing with a dictionary of lists of NumPy arrays::

        import numpy as np
        model_a = [np.random.randn(3, 3), np.random.randn(2, 2)]
        model_b = [np.random.randn(3, 3), np.random.randn(2, 2)]
        record = MultiArrayRecord({"client_1": model_a, "client_2": model_b})
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
        self, record_dict: dict[str, ArrayRecord], *, keep_input: bool = True
    ) -> None: ...

    @overload
    def __init__(
        self, numpy_ndarrays_dict: dict[str, list[NDArray]], *, keep_input: bool = True
    ) -> None: ...

    @overload
    def __init__(
        self,
        torch_state_dicts: dict[str, dict[str, torch.Tensor]] | dict[str, dict[str, Any]],
        *,
        keep_input: bool = True,
    ) -> None: ...

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *args: Any,
        record_dict: dict[str, ArrayRecord] | None = None,
        numpy_ndarrays_dict: dict[str, list[NDArray]] | None = None,
        torch_state_dicts: dict[str, dict[str, torch.Tensor]] | dict[str, dict[str, Any]] | None = None,
        keep_input: bool = True,
    ) -> None:
        super().__init__(_check_key, _check_value)

        if len(args) > 1:
            _raise_multi_array_record_init_error()
        arg = args[0] if args else None
        init_method: str | None = None

        def _try_set_arg(_arg: Any, method: str) -> None:
            if _arg is None:
                return
            nonlocal arg, init_method
            if arg is not None or init_method is not None:
                _raise_multi_array_record_init_error()
            init_method = method
            arg = _arg

        _try_set_arg(record_dict, "record_dict")
        _try_set_arg(numpy_ndarrays_dict, "numpy_ndarrays_dict")
        _try_set_arg(torch_state_dicts, "state_dicts")

        if arg is None:
            return

        # Handle dictionary of ArrayRecords
        if not init_method or init_method == "record_dict":
            if (
                isinstance(arg, dict)
                and all(isinstance(k, str) for k in arg.keys())
                and all(isinstance(v, ArrayRecord) for v in arg.values())
            ):
                record_dict = cast(dict[str, ArrayRecord], arg)
                converted = self.from_record_dict(record_dict, keep_input=keep_input)
                self.__dict__.update(converted.__dict__)
                return

        # Handle dictionary of NumPy ndarrays
        if not init_method or init_method == "numpy_ndarrays_dict":
            if isinstance(arg, dict) and all(isinstance(k, str) for k in arg.keys()):
                # Assume list of NDArrays internally check passes
                numpy_ndarrays_dict = cast(dict[str, list[NDArray]], arg)
                converted = self.from_numpy_ndarrays_dict(
                    numpy_ndarrays_dict, keep_input=keep_input
                )
                self.__dict__.update(converted.__dict__)
                return

        # Handle dictionary of PyTorch state_dicts
        if not init_method or init_method == "state_dicts":
            if (torch := sys.modules.get("torch")) is not None and isinstance(arg, dict):
                torch_state_dicts = cast(dict[str, dict[str, torch.Tensor]], arg)
                converted = self.from_torch_state_dicts(
                    torch_state_dicts, keep_input=keep_input
                )
                self.__dict__.update(converted.__dict__)
                return

        _raise_multi_array_record_init_error()

    @classmethod
    def from_record_dict(
        cls,
        record_dict: dict[str, ArrayRecord],
        *,
        keep_input: bool = True,
    ) -> MultiArrayRecord:
        """Create MultiArrayRecord from a dictionary of ArrayRecords."""
        record = MultiArrayRecord()
        for k, v in record_dict.items():
            record[k] = v  # ArrayRecord is assigned directly
        if not keep_input:
            record_dict.clear()
        return record

    @classmethod
    def from_numpy_ndarrays_dict(
        cls,
        ndarrays_dict: dict[str, list[NDArray]],
        *,
        keep_input: bool = True,
    ) -> MultiArrayRecord:
        """Create MultiArrayRecord from a dictionary of NumPy ndarrays lists."""
        record = MultiArrayRecord()
        
        for k in list(ndarrays_dict.keys()):
            v = ndarrays_dict[k] if keep_input else ndarrays_dict.pop(k)
            # Delegate memory and serialization management to ArrayRecord
            record[k] = ArrayRecord.from_numpy_ndarrays(v, keep_input=keep_input)

        if not keep_input:
            ndarrays_dict.clear()
            gc.collect()
        return record

    @classmethod
    def from_torch_state_dicts(
        cls,
        state_dicts: dict[str, dict[str, torch.Tensor]],
        *,
        keep_input: bool = True,
    ) -> MultiArrayRecord:
        """Create MultiArrayRecord from a dict of PyTorch state_dicts."""
        if "torch" not in sys.modules:
            raise RuntimeError(
                f"PyTorch is required to use {cls.from_torch_state_dicts.__name__}"
            )

        record = MultiArrayRecord()
        for k in list(state_dicts.keys()):
            v = state_dicts[k] if keep_input else state_dicts.pop(k)
            record[k] = ArrayRecord.from_torch_state_dict(v, keep_input=keep_input)

        if not keep_input:
            state_dicts.clear()
            gc.collect()
        return record

    def to_numpy_ndarrays_dict(self, *, keep_input: bool = True) -> dict[str, list[NDArray]]:
        """Return the MultiArrayRecord as a dictionary of NumPy array lists."""
        ret: dict[str, list[NDArray]] = {}
        for k in list(self.keys()):
            record_item = self[k] if keep_input else self.pop(k)
            ret[k] = record_item.to_numpy_ndarrays(keep_input=keep_input)

        if not keep_input:
            gc.collect()
        return ret

    def to_torch_state_dicts(
        self, *, keep_input: bool = True
    ) -> dict[str, OrderedDict[str, torch.Tensor]]:
        """Return the MultiArrayRecord as a dictionary of PyTorch state_dicts."""
        ret: dict[str, OrderedDict[str, torch.Tensor]] = {}
        for k in list(self.keys()):
            record_item = self[k] if keep_input else self.pop(k)
            ret[k] = record_item.to_torch_state_dict(keep_input=keep_input)
            
        if not keep_input:
            gc.collect()
        return ret

    def count_bytes(self) -> int:
        """Return number of Bytes stored in this multi-model object."""
        num_bytes = 0
        for k, v in self.items():
            num_bytes += v.count_bytes()
            num_bytes += len(k)
        return num_bytes

    @property
    def children(self) -> dict[str, InflatableObject]:
        """Return a dictionary of ArrayRecords with their Object IDs as keys."""
        return {record.object_id: record for record in self.values()}

    def deflate(self) -> bytes:
        """Deflate the MultiArrayRecord."""
        record_refs: dict[str, str] = {}
        for record_name, record in self.items():
            record_refs[record_name] = record.object_id

        # Serialize references dict
        object_body = json.dumps(record_refs).encode("utf-8")
        return add_header_to_object_body(object_body=object_body, obj=self)

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> MultiArrayRecord:
        """Inflate a MultiArrayRecord from bytes."""
        if children is None:
            children = {}

        # Inflate mapping of record_names to ArrayRecords' object IDs
        obj_body = get_object_body(object_content, cls)
        record_refs: dict[str, str] = json.loads(obj_body.decode(encoding="utf-8"))

        unique_records = set(record_refs.values())
        children_obj_ids = set(children.keys())
        if unique_records != children_obj_ids:
            raise ValueError(
                "Unexpected set of `children`. "
                f"Expected {unique_records} but got {children_obj_ids}."
            )

        # Ensure children are of type ArrayRecord
        if not all(isinstance(rec, ArrayRecord) for rec in children.values()):
            raise ValueError("`Children` are expected to be of type `ArrayRecord`.")

        return MultiArrayRecord(
            {name: cast(ArrayRecord, children[object_id]) for name, object_id in record_refs.items()}
        )

    @property
    def object_id(self) -> str:
        """Get object ID."""
        ret = super().object_id
        self.is_dirty = False
        return ret

    @property
    def is_dirty(self) -> bool:
        """Check if the object is dirty after the last deflation."""
        if "_is_dirty" not in self.__dict__:
            self.__dict__["_is_dirty"] = True

        if not self.__dict__["_is_dirty"]:
            if any(v.is_dirty for v in self.values()):
                self.__dict__["_is_dirty"] = True
        return cast(bool, self.__dict__["_is_dirty"])

    @is_dirty.setter
    def is_dirty(self, value: bool) -> None:
        """Set the dirty flag."""
        self.__dict__["_is_dirty"] = value

    def __setitem__(self, key: str, value: ArrayRecord) -> None:
        """Set item and mark the record as dirty."""
        self.is_dirty = True
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete item and mark the record as dirty."""
        self.is_dirty = True
        super().__delitem__(key)