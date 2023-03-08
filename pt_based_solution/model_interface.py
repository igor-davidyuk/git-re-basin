from collections import defaultdict
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


class ModelSupportingPermutations:
    def __init__(self, model_id: int, ) -> None:
        self.model_id = model_id
        self.model = self.get_model()
        # This is {'weight_name': (P1, P2, ...)]}
        self._weight_name_to_perm_vector = None
        # This is {'P_name': ('weight_name', target_axis)}
        self._perm_name_to_weight_names = None

    def get_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    @property
    def weight_name_to_perm_vector(self):
        """
        Get Weight Name to Permutation Vector dict.

        This method should be automated in a real product.
        """
        if self._weight_name_to_perm_vector is None:
            raise NotImplementedError

        return self._weight_name_to_perm_vector

    @property
    def perm_name_to_weight_names(self):
        "Get Permutation Name to Weight Name and Target Axis dict."
        if self._perm_name_to_weight_names is None:
            self._perm_name_to_weight_names = defaultdict(list)
            for w_name, perm_vector in self.weight_name_to_perm_vector.items():
                for axis, perm_name in enumerate(perm_vector):
                    if perm_name is not None:
                        self._perm_name_to_weight_names[perm_name].append(
                            (w_name, axis)
                        )

        return self._perm_name_to_weight_names

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

    def get_blend_model(self, admixture_model, fraction, skip_not_permuted=False):
        self_state = self.state_dict
        admixture_state = admixture_model.state_dict
        blend_state = {}
        for name in self_state:
            if name in self.weight_name_to_perm_vector:
                blend_state[name] = (
                        admixture_state[name] * fraction +
                        self_state[name] * (1 - fraction)
                    )
            else:
                if not skip_not_permuted:
                    blend_state[name] = self_state[name]
                else:
                    raise KeyError(
                        f'Permutation for weight {name} is not defined.')
        new_model = deepcopy(self)
        new_model.state_dict = blend_state
        return new_model

    def match_weights(
            self, model_b, max_iter=100, init_permutations=None, seed: int = 1
                     ) -> dict[str, np.ndarray]:
        """Find a permutation of `model_b` to make them match self weights."""
        assert isinstance(model_b, self.__class__)

        # Prepare a random number generator
        rng = np.random.default_rng(seed)

        def rngmix(seed):
            return np.random.default_rng(
                [rng._bit_generator._seed_seq.entropy, hash(seed)])

        self_state = self.state_dict
        model_b_state = model_b.state_dict
        perm_sizes = {
            perm_name: self_state[w_name_axis_list[0][0]].shape[w_name_axis_list[0][1]]
            for perm_name, w_name_axis_list in self.perm_name_to_weight_names.items()
            }

        if init_permutations is not None:
            permutations = init_permutations
        else:
            permutations = {perm_name: np.arange(perm_size)
                            for perm_name, perm_size in perm_sizes.items()}
        perm_names = list(permutations.keys())

        for iteration in range(max_iter):
            progress = False
            for p_ix in rngmix(iteration).permutation(len(perm_names)):
                perm_name = perm_names[p_ix]
                perm_size = perm_sizes[perm_name]
                A = np.zeros((perm_size, perm_size))
                # We iterate over all the (weight_name, axis) pairs
                # for a specific permutation matrix.
                for weight_name, axis in self.perm_name_to_weight_names[perm_name]:
                    w_a = self_state[weight_name]
                    w_b = self._get_permuted_tensor(permutations, weight_name, model_b_state, skip_axis=axis)
                    w_a = np.moveaxis(w_a, axis, 0).reshape((perm_size, -1))
                    w_b = np.moveaxis(w_b, axis, 0).reshape((perm_size, -1))
                    A += w_a @ w_b.T

                ri, ci = linear_sum_assignment(A, maximize=True)
                assert (ri == np.arange(len(ri))).all()

                oldL = np.vdot(A, np.eye(perm_size)[permutations[perm_name]])
                newL = np.vdot(A, np.eye(perm_size)[ci, :])
                print(f"{iteration}/{perm_name}: {newL - oldL}")
                progress = progress or newL > oldL + 1e-12

                permutations[perm_name] = np.array(ci)

            if not progress:
                break

        return permutations

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
        if weight_name not in self.weight_name_to_perm_vector:
            return weight
        perm_vector = self.weight_name_to_perm_vector[weight_name]
        if perm_vector is None:
            return weight
        for axis, perm_name in enumerate(perm_vector):
            # Skip the axis we're trying to permute.
            if axis == skip_axis:
                continue

            # None indicates that there is no permutation relevant to that axis.
            if perm_name is not None:
                print(weight_name, weight.shape, permutations[perm_name].shape)
                weight = np.take(weight, permutations[perm_name], axis=axis)

        return weight
