"""
File: logger.py
Modified by: Senthil Purushwalkam
Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
Email: spurushw<at>andrew<dot>cmu<dot>edu
Github: https://github.com/senthilps8
Description: 
"""


from tensorboard.compat.proto.summary_pb2 import HistogramProto
from torch.utils.tensorboard.writer import FileWriter
from torch.utils.tensorboard.summary import Summary
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import os
import cv2
from io import BytesIO
from utils.preprocess import get_inv_transform


class Logger(object):

    def __init__(self, log_dir, name=None):
        """Create a summary writer logging to log_dir."""
        if name is None:
            name = 'temp'
        self.name = name
        if name is not None:
            try:
                os.makedirs(os.path.join(log_dir, name))
            except:
                pass
            self.writer = FileWriter(os.path.join(log_dir, name),
                                                filename_suffix=name)
        else:
            self.writer = FileWriter(log_dir, filename_suffix=name)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step, is_image=False):
        """Log a list of images."""

        t_inv = get_inv_transform()
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            if is_image:
                im = t_inv(img).numpy().transpose(1,2,0).astype(np.uint8)
            else:
                im = img if isinstance(img, np.ndarray) else img.numpy()
                im = cv2.normalize(im,None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
            im = Image.fromarray(im)
            im.save(s, format="PNG")

            # Create an Image object
            img_sum = Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = Summary(value=[Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def to_np(self, x):
        return x.data.cpu().numpy()

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def model_param_histo_summary(self, model, step):
        """log histogram summary of model's parameters
        and parameter gradients
        """
        for tag, value in model.named_parameters():
            if value.grad is None:
                continue
            tag = tag.replace('.', '/')
            tag = self.name+'/'+tag
            self.histo_summary(tag, self.to_np(value), step)
            self.histo_summary(tag+'/grad', self.to_np(value.grad), step)

