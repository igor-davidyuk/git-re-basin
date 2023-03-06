from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch


class ModelSupportingPermutations(torch.nn.module):
    def __init__(self) -> None:
        self.model = self.get_model()
        # This is {'weight_name': (P1, P2, ...)]}
        self._weight_name_to_perm_vector = None
        # This is {'P_name': ('weight_name', target_axis)}
        self._perm_name_to_weight_name = None

    def get_model(self):
        raise NotImplementedError

    @property
    def weight_name_to_perm_vector(self):
        "Get Weight Name to Permutation Vector dict."
        if self._weight_name_to_perm_vector is None:
            raise NotImplementedError

        return self._weight_name_to_perm_vector

    @property
    def perm_name_to_weight_name(self):
        "Get Permutation Name to Weight Name and Target Axis dict."
        if self._perm_name_to_weight_name is None:
            self._perm_name_to_weight_name = defaultdict(list)
            for w_name, perm_vector in self.weight_name_to_perm_vector.items():
                for axis, perm_name in enumerate(perm_vector):
                    if perm_name is not None:
                        self._perm_name_to_weight_name[perm_name].append
                        (
                            (w_name, axis)
                        )

        return self._perm_name_to_weight_name

    @property
    def state_dict(self):
        result = {}
        for flat_name, tensor in self.model.state_dict().items():
            result[flat_name] = tensor.numpy(force=True)
        return result

    @state_dict.setter
    def state_dict(self, state_dict):
        tensor_dict = {}
        for k, v in state_dict.items():
            tensor_dict[k] = torch.tensor(v)
        self.model.load_state_dict(tensor_dict)

    def get_permuted_model(self, permutations):
        new_model = deepcopy(self)
        new_model._apply_permutation(permutations)
        return new_model

    def get_blend_model(self, admixture_model, fraction):
        self_state = self.state_dict
        admixture_state = admixture_model.state_dict
        blend_state = {}
        for name in self_state:
            blend_state[name] = (
                    admixture_state[name] * fraction +
                    self_state[name] * (1 - fraction)
                )
        new_model = deepcopy(self)
        new_model.state_dict = blend_state
        return new_model

    def _apply_permutation(self, permutations):
        weight_dict = self.state_dict
        permuted_weights_dict = {}
        for weight_name in weight_dict.keys():
            permuted_weights_dict[weight_name] = self._get_permuted_tensor(
                permutations, weight_name, weight_dict
            )
        self.state_dict = permuted_weights_dict

    def _get_permuted_tensor(
            self, permutations, weight_name, weight_dict, skip_axis=None
            ):
        weight = weight_dict[weight_name]
        perm_vector = self.weight_name_to_perm_vector[weight_name]
        if perm_vector is None:
            return weight
        for axis, perm_name in enumerate(perm_vector):
            # Skip the axis we're trying to permute.
            if axis == skip_axis:
                continue

            # None indicates that there is no permutation relevant to that axis.
            if perm_name is not None:
                weight = np.take(weight, permutations[perm_name], axis=axis)

        return weight

    def forward(self, x):
        return self.model.forward(x)
