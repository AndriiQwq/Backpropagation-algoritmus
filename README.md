# Backpropagation Algorithm for Solving XOR Problems

## Overview
This project implements a neural network to solve **XOR** and other logical problems using the **Backpropagation Algorithm**.  
The model supports flexible configurations, allowing you to set:

- Number and size of layers
- Activation functions for each layer
- Training parameters (epochs, learning rate, momentum)
- Logging options

Configuration is managed via a config file, for example:

```ini
[Training]
epoch_count = 200
learning_rate = 0.1
momentum = 0.9
use_momentum = True

[Logging]
enable_logging = True
log_level = INFO

[Network]
layer_sizes = 2,4,1
activations = Tanh,Tanh
```

### Key Features:
- **Multiple Layer Support**: Flexible architecture, set any number and size of layers.
- **Customizable Configuration**: All main parameters are set in the config file.
- **Integrated Logger**: Training and model operations are logged for easier debugging and analysis.
- **XOR Problem Solving**: The trained model achieves high accuracy on XOR and similar logical tasks.

## Conclusion
This backpropagation neural network efficiently solves logical problems like AND, OR, XOR and more.  
Optimal performance is achieved with proper configuration and logging helps