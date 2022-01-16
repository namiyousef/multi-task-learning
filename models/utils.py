from utils import _split_equation
from criterion.loss_functions import RandomCombinedLoss, SimpleCombinedLoss, DynamicCombinedLoss, NormalisedDynamicCombinedLoss
from criterion import loss_functions
from models import model
import torch

def get_prebuilt_model(encoder, decoders, losses, weights=None, apply_weights_during_test=False):
    """Function to get pre-built default models that exist in models.model
    :param encoder: name of the encoder to use
    :type encoder: str
    :param decoders: + separated names of decoder heads to use, e.g. seg+class. Currently only supports defaults.
    :type decoders: str
    :param losses: + separated names of losses to use and scaling factors, e.g. 0.01*CrossEntropyLoss+0.01*L1Loss. Prioritises PyTorch losses, if they don't exist looks for custom losses in criterion.loss_functions
    :type losses: str
    :param weights: name of weight convention to use. CAn add parameters using ::, e.g. uniform::1
    :type weights: str
    :param apply_weights_during_test: flag to determine if weights to be applied during test
    :type apply_weights_during_test: bool
    :returns: instantiated HardMTLModel and CombinedLoss
    """
    decoders = _split_equation(decoders)
    losses = _split_equation(losses, False)
    scaling_factors = {task: float(loss.split('*')[0]) if '*' in loss else 1.0 for task, loss in
                       zip(decoders, losses)}
    losses = {task: loss.split('*')[-1] for task, loss in zip(decoders, losses)}

    for task, loss_name in losses.items():
        if hasattr(torch.nn, loss_name):
            losses[task] = getattr(torch.nn, loss_name)()
        else:
            losses[task] = getattr(loss_functions, loss_name)()

    if weights is None:
        loss = SimpleCombinedLoss(losses, weights, sf=scaling_factors, eval_test=apply_weights_during_test)
    if isinstance(weights, list):
        if len(decoders) != len(weights):
            raise ValueError(
                'The number of tasks is different to the number of weights. Please make sure that decoders and weights have the same size.')
        elif sum(weights) != 1:
            raise ValueError('The sum of weights is greater than 1. Please make sure they sum up to 1.')

        loss = SimpleCombinedLoss(losses, weights, sf=scaling_factors, eval_test=apply_weights_during_test)

    elif isinstance(weights, str):
        weights = weights.split('::')
        prior = weights[0]
        if len(weights) < 2:
            raise ValueError(
                'You must include a parameter. For random loss weighting, this must be frequency. For dynamic loss weighting, this must be frequency and temperature')
        try:
            frequency = int(weights[1])
        except:
            TypeError('Frequency must be an integer')
        if prior == 'dynamic':
            temperature = float(weights[-1])
            loss = DynamicCombinedLoss(losses, frequency=frequency, temperature=temperature)
        elif prior == 'dynamic_novel':
            temperature = float(weights[-1])
            loss = NormalisedDynamicCombinedLoss(losses, frequency=frequency, temperature=temperature)
        else:
            loss = RandomCombinedLoss(losses, frequency=frequency, prior=prior, sf=scaling_factors, eval_test=apply_weights_during_test)

    decoders = sorted(decoders, reverse=True)
    net = getattr(model, f'{encoder}_{"_".join(decoders)}')()
    return net, loss