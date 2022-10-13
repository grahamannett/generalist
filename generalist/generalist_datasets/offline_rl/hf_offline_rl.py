import datasets
from torch.utils.data import Dataset

from generalist.data_types.input_types import OfflineRLType
from generalist.data_types.helper_types import SampleBuilderMixin


class GymReplay(SampleBuilderMixin):
    def __init__(self, path: str = "edbeeching/decision_transformer_gym_replay", name: str = "halfcheetah-expert-v2", split: str = "train"):
        self._path = path
        self._name = name
        self._split = split

        self._dataset = datasets.load_dataset(self._path, self._name)[self._split]

        self.sample_builder.use_preprocessing(OfflineRLType.hook_attributes.__name__)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int):
        out = self._dataset[idx]
        observations = out["observations"]
        actions = out["actions"]
        rewards = out["rewards"]
        dones = out["dones"]

        offline_rl_type = OfflineRLType(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

        sample_metadata = self.sample_builder.metadata(idx=idx, dataset_name=self.__class__.__name__)
        sample = self.sample_builder(data=offline_rl_type, metadata=sample_metadata)
        # offline_rl_type.hook_attributes(sample)
        return sample
