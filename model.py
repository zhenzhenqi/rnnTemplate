# model.py - UPDATED FOR TENSORFLOW 2.x
import tensorflow as tf
import numpy as np

# A mapping from the old model names to the new Keras layers
CELL_MAPPING = {
    'rnn': tf.keras.layers.SimpleRNN,
    'gru': tf.keras.layers.GRU,
    'lstm': tf.keras.layers.LSTM,
}

class Model(tf.keras.Model):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        # 1. Embedding Layer: Converts word indices to vectors
        self.embedding = tf.keras.layers.Embedding(args.vocab_size, args.rnn_size)

        # 2. RNN Layers: A stack of recurrent cells (LSTM, GRU, etc.)
        if args.model not in CELL_MAPPING:
            raise Exception(f"Model type not supported: {args.model}")
        
        cell_class = CELL_MAPPING[args.model]
        self.rnn_layers = []
        for i in range(args.num_layers):
            # We need to return the full sequence for all layers except the last one
            return_sequences = (i < args.num_layers - 1)
            self.rnn_layers.append(
                cell_class(args.rnn_size,
                           return_sequences=True, # Always return sequences for easier processing
                           return_state=True)
            )

        # 3. Dense Layer: The final output layer that gives logits for the vocab
        self.dense = tf.keras.layers.Dense(args.vocab_size)

    def call(self, inputs, states=None, training=False):
        """The forward pass of the model."""
        # Look up the embeddings for the input sequences
        x = self.embedding(inputs)

        if training and self.args.input_keep_prob < 1.0:
            x = tf.nn.dropout(x, 1.0 - self.args.input_keep_prob)

        # Process the sequence through all RNN layers
        new_states = []
        for i, layer in enumerate(self.rnn_layers):
            # Keras layers can take an initial state
            initial_state = states[i] if states else None
            x, *state = layer(x, initial_state=initial_state)
            new_states.append(state)

        if training and self.args.output_keep_prob < 1.0:
            x = tf.nn.dropout(x, 1.0 - self.args.output_keep_prob)
        
        # Get the final logits from the dense layer
        logits = self.dense(x)
        
        return logits, new_states

    def get_initial_state(self, batch_size):
        """Helper to create a zeroed initial state for the RNNs."""
        initial_state = []
        for layer in self.rnn_layers:
            # For LSTM, state is a list [h, c]. For GRU/SimpleRNN, it's just h.
            if isinstance(layer, tf.keras.layers.LSTM):
                initial_state.append([tf.zeros((batch_size, self.args.rnn_size)),
                                      tf.zeros((batch_size, self.args.rnn_size))])
            else:
                 initial_state.append(tf.zeros((batch_size, self.args.rnn_size)))
        return initial_state

    def sample(self, chars, vocab, num=200, prime='The ', sampling_type=1):
        """Generate text from a priming string (replaces the old sample method)."""
        # Convert the priming string to character indices
        prime_indices = [vocab[char] for char in prime]
        
        # Use the model to process the prime string and get the initial state
        input_tensor = tf.expand_dims(prime_indices, 0)
        _, state = self(input_tensor, states=self.get_initial_state(1))

        # Start generating with the last character of the prime string
        next_input = tf.expand_dims([prime_indices[-1]], 0)
        
        generated_text = ''
        for _ in range(num):
            # Get predictions for the next character
            logits, state = self(next_input, states=state)
            
            # Squeeze the unnecessary dimensions
            logits = tf.squeeze(logits, 0)
            
            # Apply sampling strategy
            if sampling_type == 0: # amax
                next_char_idx = tf.argmax(logits, axis=-1)[0]
            else: # weighted random sampling
                next_char_idx = tf.random.categorical(logits, num_samples=1)[0, 0]

            # Convert index to character and append
            pred_char = chars[next_char_idx.numpy()]
            generated_text += pred_char
            
            # The predicted character becomes the next input
            next_input = tf.expand_dims([next_char_idx], 0)
            
        return prime + generated_text