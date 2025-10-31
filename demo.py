
import os
import sys

import numpy as np
import imageio

import logging
import time

import tensorflow as tf
from convcrf import ConvCRF

from utils import pascal_visualizer as vis
from utils import synthetic

try:
    import matplotlib.pyplot as plt
    figure = plt.figure()
    matplotlib = True
    plt.close(figure)
except:
    matplotlib = False
    pass

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

def do_crf_inference(image, unary, args):

    # get basic hyperparameters
    num_classes = unary.shape[2]
    shape = image.shape[0:2]

    if args.normalize:
        # Warning, applying image normalization affects CRF computation.
        # The parameter 'col_feats::schan' needs to be adapted.

        # Normalize image range
        #     This changes the image features and influences CRF output
        image = image / 255
        # mean substraction
        #    CRF is invariant to mean subtraction, output is NOT affected
        image = image - 0.5
        # std normalization
        #       Affect CRF computation
        image = image / 0.3

        # schan = 0.1 is a good starting value for normalized images.
        # The relation is f_i = image * schan
        # config['col_feats']['schan'] = 0.1

    img_var = tf.expand_dims(tf.constant(image, tf.float32), 0) # (1, height, width, 3)
    unary_var = tf.expand_dims(tf.constant(unary, tf.float32), 0) # (1, height, width, num_classes)

    # convcrf expects logits not probs
    unary_logits = tf.math.log(tf.clip_by_value(unary_var, 1e-7, 1.0))

    # recenter image_var
    img_var = img_var - tf.reshape(
        tf.constant([123.68, 116.779, 103.939], dtype=tf.float32), # imagenet means
        (1, 1, 1, 3)
    )

    logging.info("Build ConvCRF.")
    crf = ConvCRF(
        shape,
        filter_size=7,
        blur=1,
        num_classes=num_classes,
        theta_alpha=40., # spatial
        theta_beta=3., # color
        theta_gamma=3., # spatial
        num_iterations=10,
        normalize=True,
        dtype=tf.float32,
    )

    crf.build(((None,*shape,num_classes),(None,*shape,3)))

    logging.info("Loading default weights")
    for w in crf.weights:
        if "spatial" in w.name:
            w.assign(tf.constant(np.load("./weights/spatial_kernel.npy"), tf.float32))
        elif "bilateral" in w.name:
            w.assign(tf.constant(np.load("./weights/bilateral_kernel.npy"), tf.float32))
        elif "compatibility" in w.name:
            w.assign(tf.constant(np.load("./weights/compatibility_matrix.npy"), tf.float32))
        else:
            raise ValueError("something is wrong")
    
    crf.trainable = False

    logging.info("Start Computation.")
    # Perform CRF inference
    prediction = tf.nn.softmax(crf((unary_logits, img_var)), -1)

    if args.nospeed:
        # Evaluate inference speed
        logging.info("Doing speed evaluation.")
        start_time = time.time()
        for i in range(10):
            # Running ConvCRF 10 times and average total time
            pred = crf((unary_logits, img_var))

        duration = (time.time() - start_time) * 1000 / 10

        logging.info("Finished running 10 predictions.")
        logging.info("Avg. Computation time: {} ms".format(duration))

    return prediction.numpy()


def plot_results(image, unary, prediction, label, args):

    logging.info("Plot results.")

    # Create visualizer
    myvis = vis.PascalVisualizer()

    # Transform id image to coloured labels
    coloured_label = myvis.id2color(id_image=label)

    unary_hard = np.argmax(unary, axis=2)
    coloured_unary = myvis.id2color(id_image=unary_hard)

    prediction = prediction[0]  # Remove Batch dimension (h, w, c)
    prediction_hard = np.argmax(prediction, axis=-1)
    coloured_crf = myvis.id2color(id_image=prediction_hard)

    if matplotlib:
        # Plot results using matplotlib
        figure = plt.figure()
        figure.tight_layout()

        # Plot parameters
        num_rows = 2
        num_cols = 2

        ax = figure.add_subplot(num_rows, num_cols, 1)
        # img_name = os.path.basename(args.image)
        ax.set_title('Image ')
        ax.axis('off')
        ax.imshow(image)

        ax = figure.add_subplot(num_rows, num_cols, 2)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(coloured_label.astype(np.uint8))

        ax = figure.add_subplot(num_rows, num_cols, 3)
        ax.set_title('Unary')
        ax.axis('off')
        ax.imshow(coloured_unary.astype(np.uint8))

        ax = figure.add_subplot(num_rows, num_cols, 4)
        ax.set_title('CRF Output')
        ax.axis('off')
        ax.imshow(coloured_crf.astype(np.uint8))

        plt.show()
    else:
        if args.output is None:
            args.output = "out.png"

        logging.warning("Matplotlib not found.")
        logging.info("Saving output to {} instead".format(args.output))

    if args.output is not None:
        # Save results to disk
        out_img = np.concatenate(
            (image, coloured_label, coloured_unary, coloured_crf),
            axis=1)

        imageio.imwrite(args.output, out_img.astype(np.uint8))

        logging.info("Plot has been saved to {}".format(args.output))

    return


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("image", type=str,
                        help="input image")

    parser.add_argument("label", type=str,
                        help="Label file.")

    parser.add_argument("--gpu", type=str, default='1',
                        help="which gpu to use")

    parser.add_argument('--output', type=str,
                        help="Optionally save output as img.")

    parser.add_argument('--nospeed', action='store_false',
                        help="Skip speed evaluation.")

    parser.add_argument('--normalize', action='store_true',
                        help="Normalize input image before inference.")

    parser.add_argument('--cpu', action='store_true',
                        help="Run on CPU instead of GPU.")

    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    # args = parser.parse_args()

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Load data
    image = imageio.imread(args.image)
    label = imageio.imread(args.label)

    # Produce unary by adding noise to label
    unary = synthetic.augment_label(label, num_classes=21)
    # Compute CRF inference
    prediction = do_crf_inference(image, unary, args)

    # Plot output
    plot_results(image, unary, prediction, label, args)
    logging.info("Thank you for trying ConvCRFs.")
