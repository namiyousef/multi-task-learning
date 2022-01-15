from utils import _split_equation
from criterion.loss_functions import RandomCombinedLoss, SimpleCombinedLoss, DynamicCombinedLoss
from models import bodys

def get_prebuilt_model(encoder, decoders, losses, weights=None, apply_weights_during_test=False):

    decoders = _split_equation(decoders)
    losses = _split_equation(losses)
    scaling_factors = {task: float(loss.split('*')[0]) if '*' in loss else 1.0 for task, loss in
                       zip(decoders, losses)}
    losses = {task: loss.split('*')[-1] for task, loss in zip(decoders, losses)}

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
        else:
            loss = RandomCombinedLoss(losses, frequency=frequency, prior=prior, sf=scaling_factors, eval_test=apply_weights_during_test)

    decoders = sorted(decoders)
    model = getattr(bodys, f'{encoder}_{"_".join(decoders)}')()
    return model, loss