
# setup_mnist.py
#
# Legacy AutoZOOM MNIST loader adapted for modern
# TensorFlow/Keras running in TF1-compatible graph mode.

import tensorflow as tf

# Must be executed before constructing the model.
tf.compat.v1.disable_eager_execution()

import numpy as np
import os
import gzip
import urllib.request
import h5py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)

        buf = bytestream.read(num_images * 28 * 28)

        data = np.frombuffer(
            buf,
            dtype=np.uint8
        ).astype(np.float32)

        data = (data / 255.0) - 0.5

        data = data.reshape(
            num_images,
            28,
            28,
            1
        )

        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)

        buf = bytestream.read(num_images)

        labels = np.frombuffer(
            buf,
            dtype=np.uint8
        )

    return (
        np.arange(10) == labels[:, None]
    ).astype(np.float32)


class MNIST:

    def __init__(self):

        if not os.path.exists("data"):
            os.mkdir("data")

        files = [
            "train-images-idx3-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]

        for name in files:

            path = os.path.join(
                "data",
                name
            )

            if not os.path.exists(path):

                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/cvdf-datasets/mnist/" + name,
                    path
                )

        train_data = extract_data(
            "data/train-images-idx3-ubyte.gz",
            60000
        )

        train_labels = extract_labels(
            "data/train-labels-idx1-ubyte.gz",
            60000
        )

        self.test_data = extract_data(
            "data/t10k-images-idx3-ubyte.gz",
            10000
        )

        self.test_labels = extract_labels(
            "data/t10k-labels-idx1-ubyte.gz",
            10000
        )

        VALIDATION_SIZE = 5000

        self.validation_data = train_data[
            :VALIDATION_SIZE
        ]

        self.validation_labels = train_labels[
            :VALIDATION_SIZE
        ]

        self.train_data = train_data[
            VALIDATION_SIZE:
        ]

        self.train_labels = train_labels[
            VALIDATION_SIZE:
        ]


class MNISTModel:

    def __init__(
        self,
        restore=None,
        session=None,
        use_softmax=False
    ):

        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        if session is None:
            session = tf.compat.v1.get_default_session()

        if session is None:
            raise RuntimeError(
                "MNISTModel requires an active TF session."
            )

        self.session = session

        # ---------------------------------------------------------
        # Build exactly the original architecture
        # ---------------------------------------------------------

        self.model = Sequential(
            name="mnist_classifier"
        )

        self.model.add(
            Conv2D(
                32,
                (3, 3),
                input_shape=(28, 28, 1)
            )
        )

        self.model.add(
            Activation("relu")
        )

        self.model.add(
            Conv2D(32, (3, 3))
        )

        self.model.add(
            Activation("relu")
        )

        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2)
            )
        )

        self.model.add(
            Conv2D(64, (3, 3))
        )

        self.model.add(
            Activation("relu")
        )

        self.model.add(
            Conv2D(64, (3, 3))
        )

        self.model.add(
            Activation("relu")
        )

        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2)
            )
        )

        self.model.add(
            Flatten()
        )

        self.model.add(
            Dense(200)
        )

        self.model.add(
            Activation("relu")
        )

        self.model.add(
            Dense(200)
        )

        self.model.add(
            Activation("relu")
        )

        self.model.add(
            Dense(10)
        )

        if use_softmax:

            self.model.add(
                Activation("softmax")
            )

        # Force creation of all variables.
        dummy = tf.zeros(
            [1, 28, 28, 1],
            dtype=tf.float32
        )

        self.output_tensor = self.model(dummy)

        # ---------------------------------------------------------
        # Initialize variables FIRST.
        # ---------------------------------------------------------

        self.session.run(
            tf.compat.v1.global_variables_initializer()
        )

        # ---------------------------------------------------------
        # Load legacy Keras 2.x HDF5 weights manually.
        # ---------------------------------------------------------

        if restore:
            self._load_legacy_weights(restore)

    def _load_legacy_weights(self, filename):

        print(
            "Loading legacy HDF5 weights:",
            filename
        )

        with h5py.File(filename, "r") as f:

            if "model_weights" not in f:
                raise RuntimeError(
                    "HDF5 file does not contain model_weights."
                )

            root = f["model_weights"]

            # Keras 2.x stores layer names here.
            if "layer_names" in root.attrs:

                layer_names = [
                    n.decode("utf-8")
                    if isinstance(n, bytes)
                    else n
                    for n in root.attrs["layer_names"]
                ]

            else:

                layer_names = list(root.keys())

            assignments = []

            # -----------------------------------------------------
            # Read weights layer-by-layer.
            # -----------------------------------------------------

            for layer_name in layer_names:

                if layer_name not in root:
                    continue

                layer_group = root[layer_name]

                if "weight_names" not in layer_group.attrs:
                    continue

                weight_names = [
                    n.decode("utf-8")
                    if isinstance(n, bytes)
                    else n
                    for n in layer_group.attrs["weight_names"]
                ]

                # Find corresponding Keras layer.
                keras_layer = None

                for layer in self.model.layers:

                    if layer.name == layer_name:
                        keras_layer = layer
                        break

                # Keras 3 may rename layers, so if exact
                # name is unavailable, match by having weights.
                if keras_layer is None:
                    continue

                keras_weights = keras_layer.weights

                if len(keras_weights) != len(weight_names):
                    print(
                        "WARNING:",
                        layer_name,
                        "HDF5 weights:",
                        len(weight_names),
                        "Keras weights:",
                        len(keras_weights)
                    )

                    continue

                for variable, weight_name in zip(
                    keras_weights,
                    weight_names
                ):

                    dataset_name = weight_name.split("/")[-1]

                    # Usually dataset is directly under layer group.
                    if dataset_name in layer_group:
                        value = np.array(
                            layer_group[dataset_name]
                        )
                    elif weight_name in layer_group:
                        value = np.array(
                            layer_group[weight_name]
                        )
                    else:
                        # Search recursively.
                        value = None

                        def visitor(name, obj):
                            nonlocal value

                            if (
                                value is None
                                and isinstance(
                                    obj,
                                    h5py.Dataset
                                )
                                and (
                                    name.endswith(
                                        dataset_name
                                    )
                                    or name.endswith(
                                        weight_name
                                    )
                                )
                            ):
                                value = np.array(obj)

                        layer_group.visititems(visitor)

                        if value is None:
                            raise RuntimeError(
                                "Could not find HDF5 dataset "
                                + weight_name
                            )

                    if tuple(value.shape) != tuple(
                        variable.shape
                    ):
                        raise RuntimeError(
                            "Shape mismatch for "
                            + layer_name
                            + "/"
                            + weight_name
                            + ": HDF5="
                            + str(value.shape)
                            + " Keras="
                            + str(variable.shape)
                        )

                    assignments.append(
                        variable.assign(value)
                    )

            if not assignments:

                raise RuntimeError(
                    "No weights were loaded from "
                    + filename
                )

            # Execute ALL assignments in the same session.
            self.session.run(assignments)

        print(
            "Loaded",
            len(assignments),
            "weight tensors."
        )

    def predict(self, data):
        return self.model(data)
