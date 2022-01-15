import unittest
from models.utils import get_prebuilt_model
from itertools import product, combinations, permutations

class TestGetPrebuilts(unittest.TestCase):

    def test_get_models(self):
        encoder = 'resnet34'
        decoders = ['seg', 'bb', 'class']
        configurations = []
        for r in range(1, len(decoders)+1):
            for config in permutations(decoders, r):
                configurations.append('+'.join(config))

        for config in configurations:
            get_prebuilt_model(encoder=encoder, decoders=config, losses=0)

    def test_random_weights(self):
        pass

    def test_dynamic_weights(self):
        pass

    def test_simple_weights(self):
        pass