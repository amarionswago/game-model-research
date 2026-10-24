"""Release checks with fixed tensors: no model, optimizer, or external data."""
import unittest

import torch

from swarm.population_objective import mixture, validate_members


class PublicContracts(unittest.TestCase):
    def test_uniform_mixture_matches_probability_mean(self):
        for count in (2, 4):
            probabilities = torch.tensor([[0.2, 0.8], [0.6, 0.4]] * (count // 2))
            logs, responsibilities, weights = mixture(
                probabilities.log(), torch.zeros(count, 2, 3))
            torch.testing.assert_close(logs.exp(), probabilities.mean(0))
            torch.testing.assert_close(responsibilities.sum(0), torch.ones(2))
            torch.testing.assert_close(weights, torch.full((count, 2), 1 / count))

    def test_invalid_members_and_nonfinite_inputs_reject(self):
        for members in ((0, 0), (0, 1, 2), (False, 1), (-1, 0)):
            with self.assertRaises(ValueError):
                validate_members(members)
        with self.assertRaises(FloatingPointError):
            mixture(torch.tensor([[float("nan")], [-1.0]]), torch.zeros(2, 1, 3))
        with self.assertRaises(ValueError):
            mixture(torch.zeros(2, 1), torch.zeros(4, 1, 3))


if __name__ == "__main__":
    unittest.main()
