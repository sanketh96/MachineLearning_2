import tensorflow as tf
import numpy as np

class MyDNN(tf.estimator.DNNRegressor):
    def __init__(self, feature_columns, hidden_units, savepoint):
        self.feature_columns = feature_columns
        self.hidden_units = hidden_units
        self.savepoint = savepoint

    def my_dnn_regression_fn(self, features, labels, mode, params):
        """A model function implementing DNN regression for a custom Estimator."""
        # print("here")
        # print(features)
        # print(labels)
        # Extract the input into a dense layer, according to the feature_columns.
        top = tf.feature_column.input_layer(features, params["feature_columns"])

        # Iterate over the "hidden_units" list of layer sizes, default is [20].
        i = 1
        for units in params.get("hidden_units", [20]):
            # Add a hidden layer, densely connected on top of the previous layer.
            top = tf.layers.dense(inputs = top, units = units, activation  = tf.nn.relu, name = "Layer" + str(i))
            i += 1

        # Connect a linear output layer on top.
        output_layer = tf.layers.dense(inputs = top, units = 1, name = "output_layer")

        # Reshape the output layer to a 1-dim Tensor to return predictions
        predictions = tf.squeeze(output_layer, 1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # In `PREDICT` mode we only need to return predictions.
            return tf.estimator.EstimatorSpec(
                mode = mode, predictions = {"price": predictions})

        # Calculate loss using mean squared error
        average_loss = tf.losses.mean_squared_error(labels, predictions)

        # Pre-made estimators use the total_loss instead of the average,
        # so report total_loss for compatibility.
        batch_size = tf.shape(labels)[0]
        total_loss = tf.to_float(batch_size) * average_loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = params.get("optimizer", tf.train.AdamOptimizer)
            optimizer = optimizer(params.get("learning_rate", None))
            train_op = optimizer.minimize(
                loss = average_loss, global_step = tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode = mode, loss = total_loss, train_op = train_op)

        # In evaluation mode we will calculate evaluation metrics.
        assert mode == tf.estimator.ModeKeys.EVAL

        # Calculate root mean squared error
        rmse = tf.metrics.root_mean_squared_error(labels, predictions)

        # Add the rmse to the collection of evaluation metrics.
        eval_metrics = {"rmse": rmse}

        return tf.estimator.EstimatorSpec(
            mode = mode,
            # Report sum of error for compatibility with pre-made estimators
            loss = total_loss,
            eval_metric_ops = eval_metrics)

    def train(self, input_fn, steps):
        self.model = tf.estimator.Estimator(
            model_fn = self.my_dnn_regression_fn,
            model_dir = self.savepoint,
            params = {
                "feature_columns": self.feature_columns,
                "learning_rate": 0.01,
                "optimizer": tf.train.AdamOptimizer,
                "hidden_units": self.hidden_units,
            })

        return self.model.train(input_fn = input_fn, steps = steps)

    def evaluate(self, input_fn):
        return self.model.evaluate(input_fn = input_fn)


    def get_weights(self, layer_name):
        # print(self.model.get_variable_names())
        # print("LAYER : " + layer_name)
        # print(self.model.get_variable_value(layer_name + '/kernel'))

        return self.model.get_variable_value(layer_name + '/kernel')

    def get_bias(self, layer_name):
        # print("LAYER : " + layer_name)
        # print(self.model.get_variable_value(layer_name + '/bias'))

        return self.model.get_variable_value(layer_name + '/bias')

    def getActivation(self, x):
        wts = self.get_weights("Layer1")
        bias = self.get_bias("Layer1")

        # Dot product
        a = np.matmul(x, wts) + bias

        # RELU
        if(a[0][0] <= 0):
            a[0][0] = 0
        if(a[0][1] <= 0):
            a[0][1] = 0

        return a

    def predict(self, x1, x2):
        wts = self.get_weights("output_layer").flatten()
        bias = self.get_bias("output_layer")
        x = np.array([x1, x2])
        x = x.reshape(1, 2)
        a = self.getActivation(x)

        a = a.flatten()
        return np.array([np.matmul(a, wts).sum()]) + bias

if __name__ == "__main__":
    FILE = "algebra.csv"

    def my_input_fn(file_path, perform_shuffle = False, repeat_count = 1):
        def decode_csv(line):
            parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.]])
            label = parsed_line[2] # Last element is the label
            del parsed_line[-1] # Delete last element
            del parsed_line[-1] # Delete second last element
            feature_names = ["x1", "x2"]
            features = parsed_line # Everything (but last element) are the features
            for i in range(len(features)):
                features[i] /= (np.max(features[i]) - np.min(features[i]))
            d = [dict(zip(feature_names, features)), label]
            return d

        dataset = (tf.data.TextLineDataset(file_path) # Read text file
            #.skip(1) # Skip header row
            .map(decode_csv)) # Transform each elem by applying decode_csv fn

        if perform_shuffle:
            # Randomizes input using a window of 256 elements (read into memory)
            dataset = dataset.shuffle(buffer_size = 256)

        dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
        dataset = dataset.batch(32)  # Batch size to use
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels

    fc = [tf.feature_column.numeric_column("x1"), tf.feature_column.numeric_column("x2")]
    my_model = MyDNN(feature_columns = fc, hidden_units = [2])
    '''
    next_batch = my_input_fn(FILE, True) # Will return 32 random elements

    # Now let's try it out, retrieving and printing one batch of data.
    # Although this code looks strange, you don't need to understand
    # the details.

    with tf.Session() as sess:
        first_batch = sess.run(next_batch)
    print(first_batch)
    '''
    my_model.train(input_fn = lambda: my_input_fn(FILE, True, 8), steps = 2000)
    my_model.get_weights("Layer1")
    my_model.get_weights("Layer2")
    my_model.get_weights("output_layer")
