from sacred import Ingredient
from models import __dict__

model_ingredient = Ingredient('model')
@model_ingredient.config
def config():
    arch = [
                'mobilenet',
                 'resnet18',
                 'wideres'
                ]
    num_classes = 64

@model_ingredient.capture
def get_model(
                    arch,
                    num_classes,
                    return_arch=False
                     ):
    model_sets = list()

    for k in arch:
        model_sets.append(__dict__[k](
                                    num_classes=num_classes))
    if return_arch == False:
        return model_sets
    else:
        return model_sets, arch