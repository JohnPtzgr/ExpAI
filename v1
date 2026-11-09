import tensorflow as tf
import tensorflow.python.keras.layers
from tensorflow.python.keras import layers, Model
import numpy as np
from typing import Callable, List


class QuantizedLinear(layers.Layer):
    """
    Linear layer with 2-bit weights in {-1, 0, 1, 2}
    Implements 3-group accumulation: S = -sum(G_-1) + sum(G_1) + 2*sum(G_2)

    Args:
        units: Number of output neurons
        use_ste: Whether to use Straight-Through Estimator (default: True)
        temperature: Temperature for soft quantization (default: 0.1)
        name: Layer name
    """

    def __init__(self, units, use_ste=True, temperature=0.1, name=None, primary=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.use_ste = use_ste
        self.temperature = temperature
        self.input_dim = None
        self.weight = None
        self.bias = None
        self.primary = primary

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.input_dim = input_dim

        def quantized_init(shape, dtype=None):
            # Generate random values 0-3, then map to -1,0,1,2
            if dtype:
                if self.primary is True:
                    rand_ints = tf.random.uniform(shape, minval=0, maxval=4, dtype=tf.int32)
                    # Map: 0→-1, 1→0, 2→1, 3→2
                    mapping = tf.constant([-1.0, 0.0, 1.0, 2.0], dtype=dtype)
                    return tf.gather(mapping, rand_ints)
                return tf.zeros(shape, dtype=dtype)
            if self.primary is True:
                rand_ints = tf.random.uniform(shape, minval=0, maxval=4, dtype=tf.int32)
                # Map: 0→-1, 1→0, 2→1, 3→2
                mapping = tf.constant([-1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
                return tf.gather(mapping, rand_ints)
            return tf.zeros(shape, dtype=tf.float32)

        self.weight = self.add_weight(
            name="weight",
            shape=(self.units, self.input_dim),
            initializer=quantized_init,
            trainable=True
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True
        )

        super().build(input_shape)

    @staticmethod
    def _hard_quantize(w):
        """Hard quantization to {-1, 0, 1, 2}"""
        targets = tf.constant([-1.0, 0.0, 1.0, 2.0], dtype=tf.float32)

        # Find the closest target for each weight
        w_expanded = tf.expand_dims(w, -1)  # (units, in_dim, 1)
        distances = tf.abs(w_expanded - targets)  # (units, in_dim, 4)
        indices = tf.argmin(distances, axis=-1)  # (units, in_dim)

        # Gather targets
        quantized = tf.gather(targets, indices)
        return quantized

    def _soft_quantize(self, w):
        """Soft quantization with temperature (differentiable)"""
        targets = tf.constant([-1.0, 0.0, 1.0, 2.0], dtype=tf.float32)

        # Softmax over negative distances (closer targets get higher weight)
        w_expanded = tf.expand_dims(w, -1)
        logits = -tf.abs(w_expanded - targets) / self.temperature
        probs = tf.nn.softmax(logits, axis=-1)

        # Expected value (differentiable)
        soft_quant = tf.reduce_sum(probs * targets, axis=-1)
        return soft_quant

    def _create_masks(self, w):
        """Create differentiable masks for each weight value"""
        # Smooth masks using sigmoid around boundaries
        # Boundary at -0.5, 0.5, 1.5

        # Mask for -1 (w <= -0.5)
        mask_m1 = tf.sigmoid((-w - 0.5) / self.temperature)

        # Mask for 0 (-0.5 < w < 0.5)
        mask_0 = tf.sigmoid((-w - 0.5) / self.temperature) * tf.sigmoid((w - 0.5) / self.temperature)

        # Mask for 1 (0.5 < w < 1.5)
        mask_1 = tf.sigmoid((w - 0.5) / self.temperature) * tf.sigmoid((1.5 - w) / self.temperature)

        # Mask for 2 (w > 1.5)
        mask_2 = tf.sigmoid((w - 1.5) / self.temperature)

        return mask_m1, mask_0, mask_1, mask_2

    def call(self, inputs, training=None, masks=None):
        """
        Forward pass

        Args:
            inputs: Shape (batch, input_dim)
            training: Whether in training mode
            masks: optional pre-computed (mask_m1, mask_1, mask_2) for inference

        Returns:
            outputs: Shape (batch, units), values in [-32768, 32767]
        """
        w = self.weight

        if masks:
            mask_m1, _, mask_1, mask_2 = masks

        # mask_m1, mask_0, mask_1, mask_2 = None, None, None, None

        elif self.use_ste and training:
            # Training mode: use soft quantization with temperature
            # This allows gradients to flow
            w_quantized = self._soft_quantize(w)

            # STE: forward uses soft quantized, backward uses straight-through
            w_effective = tf.stop_gradient(w_quantized - w) + w

            # Create differentiable masks
            mask_m1, _, mask_1, mask_2 = self._create_masks(w_effective)

        else:
            # Inference mode: use hard quantization
            w_quantized = self._hard_quantize(w)
            w_effective = w_quantized

            # Hard masks
            mask_m1 = tf.cast(tf.equal(w_effective, -1.0), tf.float32)
            mask_1 = tf.cast(tf.equal(w_effective, 1.0), tf.float32)
            mask_2 = tf.cast(tf.equal(w_effective, 2.0), tf.float32)
            # mask_0 = tf.cast(tf.equal(w_effective, 0.0), tf.float32)

        # 3-group accumulation
        # Inputs: (batch, in_dim)
        # Masks: (units, in_dim)
        # Output: (batch, units)
        sum_m1 = tf.matmul(inputs, mask_m1, transpose_b=True)
        sum_1 = tf.matmul(inputs, mask_1, transpose_b=True)
        sum_2 = tf.matmul(inputs, mask_2, transpose_b=True)

        # Weighted sum: s = -sum_m1 + sum_1 + 2*sum_2
        s = -sum_m1 + sum_1 + 2 * sum_2

        # Count non-zero weights per neuron
        n_nonzero = tf.reduce_sum(mask_1 + mask_m1 + mask_2, axis=1)  # (units,)
        n_nonzero = tf.maximum(n_nonzero, 1.0)  # Avoid division by zero
        n_nonzero = tf.expand_dims(n_nonzero, axis=0)  # (1, units)

        # Normalize and add bias
        normalized = (s + tf.expand_dims(self.bias, axis=0)) / n_nonzero

        # Scale to 16-bit range
        output = tf.tanh(normalized) * 32767.0
        output = tf.clip_by_value(output, -32767.0, 32767.0)

        return output

    def get_quantized_weights(self):
        """Return hard quantized weights for analysis"""
        return self._hard_quantize(self.weight)

    def get_weight_stats(self):
        """Print weight distribution statistics"""
        w = self.get_quantized_weights()
        total = tf.size(w)

        counts = {
            -1: tf.reduce_sum(tf.cast(tf.equal(w, -1.0), tf.int32)),
            0: tf.reduce_sum(tf.cast(tf.equal(w, 0.0), tf.int32)),
            1: tf.reduce_sum(tf.cast(tf.equal(w, 1.0), tf.int32)),
            2: tf.reduce_sum(tf.cast(tf.equal(w, 2.0), tf.int32)),
        }

        print(f"    Total weights: {total.numpy():,}")
        for val in [-1, 0, 1, 2]:
            count = counts[val].numpy()
            pct = 100.0 * count / total.numpy()
            print(f"    {val:2d}: {pct:5.1f}%")

        return counts


class Float32Linear(layers.Layer):
    """
    Traditional float32 linear layer for final classification.
    Takes 16-bit quantized inputs and produces float32 logits.


    Args:
        units: Number of output neurons
        use_batch_norm: Whether to use Batch Normalization (default: True)
        activation: Activation function (default: sigmoid)
        name: Layer name
    """

    def __init__(self,
                 units,
                 use_batch_norm=True,
                 activation=tf.sigmoid,
                 name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.use_batch_norm = use_batch_norm
        self.input_dim = 0
        self.weight = None
        self.bias = None
        self.activation = activation
        self.batch_norm = None

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        # Standard float32 weights (normal initialization)
        self.weight = self.add_weight(
            name="weight",
            shape=(self.units, self.input_dim),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            dtype=tf.float32
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer=tf.zeros_initializer(),
            trainable=True,
            dtype=tf.float32
        )

        if self.use_batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization()

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch, input_dim) in 16-bit integer range [-32768, 32767]
            training: training mode flag

        Returns:
            logits: (batch, units) float32 logits
        """
        # Normalize inputs from 16-bit range to [-1, 1]
        x = inputs / 32767.0

        # Standard linear transformation
        x = tf.matmul(x, self.weight, transpose_b=True)
        x = x + self.bias

        if self.activation:
            x = self.activation(x)

        # Optional batch normalization
        if self.use_batch_norm:
            x = self.batch_norm(x, training=training)

        return x

    def get_weight_stats(self):
        """Print weight statistics"""
        w = self.weight.numpy()
        print(f"    Total weights: {w.size:,}")
        print(f"    Mean: {w.mean():.4f}, Std: {w.std():.4f}")
        print(f"    Min: {w.min():.4f}, Max: {w.max():.4f}")


class SkipConnection(layers.Layer):
    """
    Skip connection with configurable length and decay

    Args:
        source_dim: Source layer dimensions
        target_dim: Target layer dimensions
        skip_length: The length of the skip connections or one more than how many layers does it bypass
    """

    def __init__(self, source_dim, target_dim, skip_length, decay_weight=1.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.skip_length = skip_length
        self.decay_weight = decay_weight

        # Linear projection for skip connection (starts at zero)
        if skip_length > 2:
            self.linear = QuantizedLinear(target_dim, primary=False, name=f"{name}_proj")
        else:
            self.linear = QuantizedLinear(target_dim, primary=True, name=f"{name}_proj")

    def build(self, input_shape):
        self.linear.build((None, self.source_dim))
        super().build(input_shape)

    def call(self, source_output, training=None, masks=None):
        """
        source_output: from skip_length layers back
        """
        proj_output = self.linear(source_output, training=training, masks=masks)
        # Apply decay weight (in hardware: right shift)
        return proj_output * self.decay_weight

    def get_masks(self):
        return self.linear.get_masks()

    def get_weight_stats(self):
        print(f"    Skip length {self.skip_length} (decay: {self.decay_weight}):")
        self.linear.get_weight_stats()


class QuantizedNetwork(Model):
    """
    Configurable quantized network with float32 final layer

    Features:
    - Quantized hidden layers with 2-bit weights
    - Float32 final layer for classification
    - Configurable decay function for skip connections
    - Progressive training: train one skip length at a time
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            hidden_activations: List[Callable],
            output_dims: int,
            decay_function: Callable[[int], float] = None,
            temperature: float = 0.1,
            use_ste: bool = True,
            use_batch_norm: bool = True,
            final_activation: Callable = tf.sigmoid,
            name: str = "quantized_network"
    ):
        super().__init__(name=name)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.output_dims = output_dims
        self.temperature = temperature
        self.use_ste = use_ste
        self.hidden_activations = hidden_activations

        # Default decay: paired half (1, 1, 1/2, 1/2, 1/4, 1/4, ...)
        if decay_function is None:
            self.decay_function = self._paired_half_decay
        else:
            self.decay_function = decay_function

        # Calculate maximum skip depth based on 16-bit precision
        self.max_skip_depth = self._calculate_max_depth()
        print(f"Maximum skip depth: {self.max_skip_depth}")

        # Dimensions for each layer (0 = input)
        self.dimensions = [input_dim] + hidden_dims

        # Create quantized hidden layers
        self.hidden_layers = []
        for i, dim in enumerate(hidden_dims):
            prev_dim = self.dimensions[i]
            layer = QuantizedLinear(
                units=dim,
                use_ste=use_ste,
                temperature=temperature,
                primary=True,
                name=f"hidden_{i}"
            )
            layer.build((1, self.dimensions[i]))
            self.hidden_layers.append(layer)

        # Create skip connections
        self.skip_connections = {}
        for skip_len in range(2, self.max_skip_depth + 1):
            for target_idx in range(skip_len, self.num_layers + 1):
                source_idx = target_idx - skip_len
                source_dim = self.dimensions[source_idx]
                target_dim = self.dimensions[target_idx]
                decay = self.decay_function(skip_len)

                # Only create if decay is non-zero
                if decay * (1 << 16) > 1:
                    skip = SkipConnection(
                        source_dim=source_dim,
                        target_dim=target_dim,
                        skip_length=skip_len,
                        decay_weight=decay,
                        name=f"skip_{skip_len}_{source_idx}_{target_idx}"
                    )
                    skip.build((1, source_dim))
                    key = (skip_len, source_idx, target_idx)
                    self.skip_connections[key] = skip

        # Float32 final layer
        self.final_layer = Float32Linear(
            units=output_dims,
            use_batch_norm=use_batch_norm,
            activation=final_activation,
            name="final",
        )
        self.final_layer.build((1, self.dimensions[self.num_layers]))

        # Training state: which skip length is currently trainable
        self.active_skip_length = 1  # Start with only direct connections

    @staticmethod
    def _paired_half_decay(skip_len: int) -> float:
        """Paired half decay: 1, 1, 1/2, 1/2, 1/4, 1/4, ..."""
        if skip_len <= 2:
            return 1.0
        else:
            exponent = (skip_len - 1) // 2
            return 2.0 ** (-exponent)

    def _calculate_max_depth(self) -> int:
        """Find maximum skip depth where decay >= 2^(-15)"""
        min_depth = 1
        max_depth = 2

        while True:
            decay = self.decay_function(max_depth)
            if decay < 2.0 ** (-15):
                break
            else:
                min_depth = max_depth
                max_depth *= 2

        while True:
            if self.decay_function((min_depth + max_depth) // 2) < 2.0 ** (-15):
                min_depth = (min_depth + max_depth) // 2
            else:
                max_depth = (min_depth + max_depth) // 2
            if max_depth - min_depth < 2:
                break

        depth = min_depth
        while True:
            decay = self.decay_function(depth + 1)
            if decay < 2.0 ** (-15):
                break
            depth += 1
        return depth

    def set_active_skip_length(self, skip_len: int):
        """
        Set which skip length to train.
        Only connections of this length will receive gradients.
        """
        self.active_skip_length = skip_len

        # self.trainable_variables.clear()

        # Freeze/unfreeze skip connections
        for (s_len, _, _), skip in self.skip_connections.items():
            if s_len == skip_len:
                skip.linear.trainable = True
                # self.trainable_variables.extend(skip.linear.trainable_variables)
            else:
                skip.linear.trainable = False


        # Hidden layers are always trainable
        for ilayer in self.hidden_layers:
            ilayer.trainable = True
            # self.trainable_variables.extend(ilayer.trainable_variables)

        # Final layer is always trainable
        self.final_layer.trainable = True
        # self.trainable_variables.extend(self.final_layer.trainable_variables)

    def call(self, inputs, training=None, return_activations=False):
        """
        Forward pass

        Args:
            inputs: (batch, input_dim)
            training: training mode flag
            return_activations: if True, return all hidden activations

        Returns:
            logits: (batch, output_dims) float32 logits
        """
        x = inputs
        x = tf.reshape(x, (-1, self.input_dim))

        # Store outputs for skip connections
        outputs = [x]  # output[0] is input

        # Process each hidden layer
        for i, layer in enumerate(self.hidden_layers):
            # Start with previous layer output
            current = layer(outputs[-1], training=training)

            # Add skip connections from all lengths
            for skip_len in range(2, self.max_skip_depth + 1):
                source_idx = i + 1 - skip_len
                if source_idx >= 0:
                    key = (skip_len, source_idx, i + 1)
                    if key in self.skip_connections:
                        skip = self.skip_connections[key]
                        skip_out = skip(outputs[source_idx], training=training)

                        # During training with progressive isolation:
                        # Only the active skip length contributes to gradient
                        if training and self.active_skip_length != skip_len:
                            skip_out = tf.stop_gradient(skip_out)

                        current = current + skip_out


            # Apply activation
            current = self.hidden_activations[i](current)
            # current = tf.nn.relu(current)
            outputs.append(current)

        # Final float32 layer
        logits = self.final_layer(outputs[-1], training=training)

        if return_activations:
            return logits, outputs[1:]  # Skip input

        return logits

    def train_skip_length(self, skip_len: int, epochs: int, train_dataset, val_dataset, optimizer, verbose=True):
        """
        Train a specific skip length while freezing others

        Args:
            skip_len: which skip length to train (1 = direct connections, 2, 3, ...)
            epochs: number of epochs for this stage
            train_dataset: tf.data.Dataset for training
            val_dataset: tf.data.Dataset for validation
            optimizer: TensorFlow optimizer
            verbose: print progress
        """
        self.set_active_skip_length(skip_len)
        # if hasattr(optimizer, 'build'):
        #     optimizer.build(self.trainable_variables)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Training skip length {skip_len} (decay: {self.decay_function(skip_len):.6f})")
            print(f"{'=' * 60}")

        # Metrics
        train_loss_metric = tf.keras.metrics.Mean()
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        for epoch in range(epochs):
            # Training
            train_loss_metric.reset_state()
            train_acc_metric.reset_state()

            cnt = 0
            i = 0
            for batch_x, batch_y in train_dataset:
                cnt += 1
                loss = None
                with tf.GradientTape() as tape:
                    logits = self(batch_x, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, logits, from_logits=True)
                    loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))

                train_loss_metric.update_state(loss)
                train_acc_metric.update_state(batch_y, logits)
                if verbose and cnt >= len(train_dataset) / 11 and i < 10:
                    i += 1
                    print(f"\r Epoch {epoch + 1}: (",
                          cnt + (i * len(train_dataset) // 11),
                          "/",
                          len(train_dataset),
                          ") [" + ("=" * i) + ("-" * (10 - i)) + f"] Loss: {train_loss_metric.result():.4f}",
                          flush=True,
                          end='')
                    cnt = 0
            print()
            # Validation
            val_acc_metric.reset_state()
            for batch_x, batch_y in val_dataset:
                logits = self(batch_x, training=False)
                val_acc_metric.update_state(batch_y, logits)

            if verbose:
                print(f"  Epoch {epoch + 1:2d}: "
                      f"Loss = {train_loss_metric.result():.4f}, "
                      f"Train Acc = {train_acc_metric.result():.4f}, "
                      f"Val Acc = {val_acc_metric.result():.4f}")
                self.get_weight_stats()

        return train_acc_metric.result().numpy(), val_acc_metric.result().numpy()

    def progressive_train(self, stages: List[int], epochs_per_stage: int, train_dataset, val_dataset, optimizer):
        """
        Progressive training: train each skip length sequentially

        Args:
            stages: list of skip lengths to train (e.g., [1, 2, 3, 4])
            epochs_per_stage: number of epochs per stage
            train_dataset: tf.data.Dataset for training
            val_dataset: tf.data.Dataset for validation
            optimizer: TensorFlow optimizer
        """
        results = {}

        for skip_len in stages:
            train_acc, val_acc = self.train_skip_length(
                skip_len=skip_len,
                epochs=epochs_per_stage,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                optimizer=optimizer,
                verbose=True
            )
            results[skip_len] = {'train_acc': train_acc, 'val_acc': val_acc}

        return results

    def get_weight_stats(self):
        """Print weight distribution for all layers"""
        print("\n" + "=" * 60)
        print("Weight Distribution Statistics:")
        print("=" * 60)

        print("\n--- Quantized Hidden Layers ---")
        for i, layer in enumerate(self.hidden_layers):
            print(f"\nhidden_{i}:")
            layer.get_weight_stats()

        print("\n--- Skip Connections ---")
        for key, skip in self.skip_connections.items():
            print(f"\nskip_{key}:")
            skip.get_weight_stats()

        print("\n--- Float32 Final Layer ---")
        self.final_layer.get_weight_stats()
        print("=" * 60 + "\n")

    def get_inference_masks(self):
        """Get all masks for hardware inference"""
        masks = {}

        for i, layer in enumerate(self.hidden_layers):
            masks[f"hidden_{i}"] = layer.get_masks()

        for key, skip in self.skip_connections.items():
            masks[f"skip_{key}"] = skip.get_masks()

        return masks


# import tensorflow as tf


# Create dataset
def load_mnist(batch_size=128):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32)
    x_train = (x_train / 127.5) - 1.0  # Map [0,255] to [-1, 1]
    x_train = x_train * 32767  # Scale to 16-bit range
    x_test = x_test.reshape(-1, 784).astype(np.float32)
    x_test = (x_test / 127.5) - 1.0
    x_test = x_test * 32767
    # x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    # x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


# Create network
network = QuantizedNetwork(
    input_dim=784,
    hidden_dims=[256, 128, 64],
    hidden_activations=[tf.nn.relu] * 3,
    output_dims=10,
    decay_function=None,  # Use default paired half
    temperature=0.5,
    use_ste=True
)

# Load data
train_ds, test_ds = load_mnist()

# Optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# Progressive training
results = network.progressive_train(
    stages=[1, 2, 3],
    epochs_per_stage=5,
    train_dataset=train_ds,
    val_dataset=test_ds,
    optimizer=optimizer
)

# Show final weight distribution
network.get_weight_stats()
