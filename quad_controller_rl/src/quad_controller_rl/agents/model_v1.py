from tensorflow.keras import layers, models, optimizers, regularizers
import tensorflow.keras.backend as K


class Actor():
    """ Actor policy model """
    def __init__(self, state_size, action_size, action_low, action_high):
      """initialize the parameters and model"""

      self.state_size = state_size
      self.action_size = action_size
      self.action_low = action_low
      self.action_high = action_high
      self.action_range = self.action_high - self.action_low

      self.build_model()

    def build_model(self):
      """ mapping of states to actions """
      # defining input layer state
      states = layers.Input(shape=(self.state_size ,), name = 'states')

      # adding hidden layers
      net = layers.Dense(units = 64, use_bias = False, kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01))(states)
      net = layers.BatchNormalization()(net)
      net = layers.Activation('relu')(net)
      net = layers.Dropout(0.5)(net)

      net = layers.Dense(units = 64, use_bias = False, kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01))(net)
      net = layers.BatchNormalization()(net)
      net = layers.Activation('relu')(net)
      net = layers.Dropout(0.5)(net)

    #   net = layers.Dense(units = 128, use_bias = False, kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01))(net)
    #   net = layers.BatchNormalization()(net)
    #   net = layers.Activation('relu')(net)
    #   net = layers.Dropout(0.5)(net)

    #   net = layers.Dense(units = 64, use_bias = False, kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01))(net)
    #   net = layers.BatchNormalization()(net)
    #   net = layers.Activation('relu')(net)
    #   net = layers.Dropout(0.5)(net)

      # output_layer
      raw_actions = layers.Dense(units = self.action_size, activation = 'tanh', name = 'raw_actions')(net)

      actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name = 'actions')(raw_actions)

      # keras model
      self.model = models.Model(inputs = states, outputs = actions)

      # loss function using action value (Q value) gradients
      action_gradients = layers.Input(shape=(self.action_size,))
      loss = K.mean(-action_gradients * actions)

      # Incorporate any additional losses here (e.g. from regularizers)

      # Define optimizer and training function
      optimizer = optimizers.Adam()
      updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
      self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], outputs=[], updates=updates_op)

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=64, use_bias = False, kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dropout(0.5)(net_states)

        net_states = layers.Dense(units=64, use_bias = False, kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dropout(0.5)(net_states)

        # net_states = layers.Dense(units=64, use_bias = False, kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01))(states)
        # net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Activation('relu')(net_states)
        # net_states = layers.Dropout(0.5)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=64, use_bias = False, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dropout(0.5)(net_actions)

        net_actions = layers.Dense(units=64, use_bias = False, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dropout(0.5)(net_actions)

        # net_actions = layers.Dense(units=64, use_bias = False, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(net_actions)
        # net_actions = layers.BatchNormalization()(net_actions)
        # net_actions = layers.Activation('relu')(net_actions)
        # net_actions = layers.Dropout(0.5)(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()],outputs=action_gradients)
