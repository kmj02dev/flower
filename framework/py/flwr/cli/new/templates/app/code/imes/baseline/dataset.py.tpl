"""$project_name: A Flower / $framework_str app."""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from datasets import load_dataset  # 전체 데이터셋 로드용
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner, DirichletPartitioner

# --------------------------------------------------------------------------
# 1. Configuration & Metadata Mapping
# --------------------------------------------------------------------------
# 데이터셋별로 달라지는 컬럼명, 테스트 스플릿, 채널 수, 정규화 수치를 관리합니다.
DATASET_META = {
    "uoft-cs/cifar10": {
        "img_col": "img", "label_col": "label", "test_split": "test",
        "channels": 3, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)
    },
    "cifar100": {
        "img_col": "img", "label_col": "fine_label", "test_split": "test",
        "channels": 3, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)
    },
    "zh-plus/tiny-imagenet": {
        "img_col": "image", "label_col": "label", "test_split": "valid",
        "channels": 3, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)
    },
    "mnist": {
        "img_col": "image", "label_col": "label", "test_split": "test",
        "channels": 1, "mean": (0.5,), "std": (0.5,)
    },
}

# --------------------------------------------------------------------------
# 2. Global Variables for Caching
# --------------------------------------------------------------------------
_CACHE = {
    "fds": None,
    "cds": None,
    "dataset_name": None
}

# --------------------------------------------------------------------------
# 3. Helper Functions
# --------------------------------------------------------------------------

def dict_collate_fn(batch):
    """DataLoader에서 {'X': tensor, 'y': tensor} 형태로 배치를 반환"""
    return {
        "X": torch.stack([item["X"] for item in batch]),
        "y": torch.tensor([item["y"] for item in batch])
    }

def apply_transforms(batch, dataset_name: str):
    """메타데이터를 참조하여 동적으로 Transform을 생성하고 적용합니다."""
    meta = DATASET_META.get(dataset_name)
    if not meta:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    img_col = meta["img_col"]
    label_col = meta["label_col"]
    
    # 데이터셋에 맞는 Transform 생성
    transforms = Compose([
        ToTensor(), 
        Normalize(meta["mean"], meta["std"])
    ])

    # 3채널(RGB)과 1채널(L: Grayscale) 구분하여 변환
    img_mode = "RGB" if meta.get("channels", 3) == 3 else "L"
    
    batch["X"] = [transforms(img.convert(img_mode)) for img in batch[img_col]]
    batch["y"] = batch[label_col]
    
    return batch

def _create_partitioner(num_partitions: int, distribution: str, partition_by: str, num_classes_per_partition: int|None = None, alpha: int|None = None, **kwargs):
    """설정에 맞는 Partitioner 객체를 생성하는 팩토리 함수"""
    if distribution == "iid":
        return IidPartitioner(num_partitions=num_partitions)
    if distribution == "pathological":
        return PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by=partition_by,
            num_classes_per_partition=num_classes_per_partition,
        )
    if distribution == "dirichlet":
        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=partition_by,
            alpha=alpha,
        )
    raise ValueError(f"Unknown distribution: {distribution}")

# --------------------------------------------------------------------------
# 4. Main Functions
# --------------------------------------------------------------------------

def load_data(partition_id: int, num_partitions: int, batch_size: int, **kwargs):
    """클라이언트용 데이터 로드 함수"""
    dataset_name = kwargs.get("dataset", "uoft-cs/cifar10")
    test_size = kwargs.get("test-size", 0.2)
    seed = kwargs.get("seed", 42)

    # 1. FederatedDataset 초기화 및 캐싱
    if _CACHE["fds"] is None or _CACHE["dataset_name"] != dataset_name:
        partitioner = _create_partitioner(
            num_partitions, **kwargs
        )
        _CACHE["fds"] = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
        _CACHE["dataset_name"] = dataset_name

    # 2. 파티션 로드 및 분할
    partition = _CACHE["fds"].load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=test_size, seed=seed)

    # 3. Transform 적용
    partition_train_test = partition_train_test.with_transform(
        lambda batch: apply_transforms(batch, dataset_name)
    )

    # 4. DataLoader 반환
    loaders = {
        split: DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            collate_fn=dict_collate_fn, num_workers=2
        )
        for split, dataset in partition_train_test.items()
    }
    
    return loaders["train"], loaders["test"]

def get_global_testloader(batch_size: int, dataset_name: str = "uoft-cs/cifar10"):
    """서버 또는 클라이언트 평가용 Global Test DataLoader 생성"""
    meta = DATASET_META.get(dataset_name)
    if not meta:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 1. 캐싱된 데이터가 없으면 순수 HuggingFace datasets로 직접 로드
    if _CACHE["cds"] is None or _CACHE["dataset_name"] != dataset_name:
        print(f"Loading global test dataset: {dataset_name}...")
        
        # HF datasets.load_dataset을 사용하여 곧바로 로드
        raw_dataset = load_dataset(dataset_name, split=meta["test_split"])
        
        # Transform 적용
        _CACHE["cds"] = raw_dataset.with_transform(
            lambda batch: apply_transforms(batch, dataset_name)
        )
        _CACHE["dataset_name"] = dataset_name
        print("Global test dataset loaded and cached.")

    # 2. DataLoader 반환
    return DataLoader(
        _CACHE["cds"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dict_collate_fn,
        num_workers=2
    )