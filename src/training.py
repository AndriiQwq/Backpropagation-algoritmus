import numpy as np

def train_model(model, config):
    """Training data"""
    training_data = [
        (np.array([0, 0]).reshape(1, -1), np.array([0]).reshape(1, -1)),
        (np.array([0, 1]).reshape(1, -1), np.array([1]).reshape(1, -1)),
        (np.array([1, 0]).reshape(1, -1), np.array([1]).reshape(1, -1)),
        (np.array([1, 1]).reshape(1, -1), np.array([0]).reshape(1, -1))
    ]
    training_data_size = len(training_data)
    losses = []

    for epoch in range(config.epoch_count):
        """Reshuffle the training data"""
        np.random.shuffle(training_data)

        total_error = 0

        for i in range(training_data_size):
            model.set_X(training_data[i][0])
            model.set_Y(training_data[i][1])

            model.forward(training_data[i][0])

            error = model.get_last_layer_error()
            model.backward(error)

            total_error += model.MSE_Loss_evaluating()


        average = total_error / len(training_data)
        losses.append(total_error / len(training_data))

        """Check stopping condition"""
        if average < 0.00001 or epoch == 20000:
            print("We reached the optimal model")
            break

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}/{config.epoch_count}, Loss: {average:.5f}")

    return losses