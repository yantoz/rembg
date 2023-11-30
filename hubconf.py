dependencies = ['torch', 'numpy', 'onnxruntime', 'pooch', 'pymatting', 'jsonschema']

import os
import torch
import numpy as np

from rembg import new_session, remove

import logging

#logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("rembg")

class _rembg():

    def __init__(self, progress=True, map_location=None):
        self.progress = progress
        self.map_location = map_location

    def __call__(self, image, mask=False, rgbamask=False, model="u2net", **kwargs):

        log.debug("Input: {}".format(image.shape))
        log.debug("Mask: {}".format(mask))

        # temporarily remove alpha channel
        alpha = None
        h, w, d = image.shape
        if d == 4:
            image, alpha = image[:, :, 0:3], image[:, :, 3:4]

        # remove background
        U2NET_HOME = os.path.join(torch.hub.get_dir(), "checkpoints", "rembg")
        os.makedirs(U2NET_HOME, exist_ok=True)
        os.environ["U2NET_HOME"] = U2NET_HOME
        if self.map_location == torch.device('cpu'):
            providers = ['CPUExecutionProvider']
        else:
            providers = None
        session = new_session(model_name=model, providers=providers)
        output = remove(image, session=session, **kwargs)

        # combine alpha channel
        if not alpha is None:
            output_alpha = output[:, :, 3:4]
            output_alpha = np.minimum(output_alpha, alpha)
            output = np.concatenate([output[:, :, 0:3], output_alpha], axis=2)

        if mask:
            output = output[:, :, 3:4]
            if rgbamask:
                output = np.concatenate([output, output, output, np.ones_like(output)*255], axis=2)

        log.debug("Output: {}".format(image.shape))

        return output

def RemBg(progress=True, map_location=None):
    return _rembg(progress, map_location)
