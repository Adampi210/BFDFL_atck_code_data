import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration
# Always check first if GPU is avaialble
device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If CUDA is not avaialbe, print message that CPU will be used
if device_used != torch.device('cuda'):
    print(f'CUDA not available, have to use {device_used}')

# Next, create a class for the neural net that will be used

class NetBasic(nn.Module):
    def __init__(self):
        # Inherit the __init__() method from nn.Module (call __init__() from the parent class of NetBasic)
        super(NetBasic, self).__init__()

        # Define layers of the NetBasic
        # Add first convolutional layer (stride specifies to move kernel by 1 pixel horizontally and vertically, padding = 'same' specifies to pad the output with 0s to preserve image size (decreases in convolution))
        self.conv0 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        # Add Batch Normalization layer (Basically normalizes batch by subtracting mean and dividing by st.dev.)
        # num_features = input channels, eps = scalable epsilon value used in normalization formula, momentum adds moving average (takes 0.1 * prev input + 0.9 * new)
        # And affine allows to learn the gamma and beta parameters used in batch normalization
        self.bn0   = nn.BatchNorm2d(num_features = 16, momentum = 0.1, eps = 1e-5, affine = True)
        # Add relu activation layer
        self.relu0 = nn.ReLU()
        # Add max pooling layer
        self.pool0 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # Next add another layers batch: convolutional, batch normalization, relu, max pooling
        self.conv1 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.bn1   = nn.BatchNorm2d(num_features = 32, momentum = 0.1, eps = 1e-5, affine = True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # And another layers batch, similar to previous, no pooling this time
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.bn2   = nn.BatchNorm2d(num_features = 64, momentum = 0.1, eps = 1e-5, affine = True)
        self.relu2 = nn.ReLU()

        # Finally add last layers batch, that includes flattening the output, and 2 linear layers (only add linear here)
        self.lin0  = nn.Linear(in_features = 64 * 7 * 7, out_features = 128)
        self.lin1  = nn.Linear(in_features = 128, out_features = 10)

    # Push the input through the net
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = torch.flatten(x, 1) # Flatten the tensor, except the batch dimension
        x = self.lin0(x)
        x = self.lin1(x)

        return x

# Create a compiled model class (or learner class) that accepts a created model, optimizer, and loss function, and gives some training/testng functionality
class CompiledModel():
    def __init__(self, model, optimizer, loss_func):
        self.model = model          # Set the class model attribute to passed mode
        self.optimizer = optimizer  # Set optimizer attribute to passed optimizer
        self.loss_func = loss_func  # Set loss function attribute to passed loss function
    
    # Training method
    def train(self, train_data, train_labels, train_batch_size, n_epoch):
        self.model.train() # Put the model in training mode

        # Create a dataloader for training
        data   = train_data.clone().detach()    # Represent data as a tensor
        labels = train_labels.clone().detach()  # Represent labels as a tensor
        train_dataset = torch.utils.data.TensorDataset(data, labels) # Create the training dataset
        # Create the train dataloader
        train_dataloader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True)

        # Next fit the model by running the training loop
        # Run for specified number of epochs
        for epoch in range(n_epoch):
            # Then for each batch of specified batch size, train the model
            for x, y in train_dataloader:
                self.optimizer.zero_grad()       # Reset the gradients
                output = self.model(x)           # Calculate the outputs for batch inputs
                loss = self.loss_func(output, y) # Calculate the loss
                loss.backward()                  # Calculate gradients
                self.optimizer.step()            # Update the parameters
                # Print batch loss  
                print(f'Epoch: {epoch}, Batch loss: {loss}')

    # Validate the model method
    def validate(self, valid_data, valid_labels):
        self.model.eval() # Set model in the evaluation mode
        
        # Create validation dataloader
        data = valid_data.clone().detach()     # Get validation data
        labels = valid_labels.clone().detach() # Get validation labels
        valid_dataset = torch.utils.data.TensorDataset(data, labels) # Create the validation dataset
        # Create validation dataloader
        valid_dataloader = DataLoader(dataset = valid_dataset, shuffle = True)

        # Initialize validation parameters
        total_loss    = 0 # Total loss measured
        total_correct = 0 # Total correct predictions
        total_pts     = len(valid_dataloader.dataset) # Total number of all points
        with torch.no_grad():
            for x, y in valid_dataloader:
                output = self.model(x)                                 # Predict the output
                total_loss += self.loss_func(output, y)                # Calculate loss for given datapoint and increment total loss
                total_correct += (torch.argmax(output).item() == y.item())

        # Find loss and accuracy
        valid_loss  = total_loss / total_pts
        valid_accur = total_correct / total_pts

        # Set model back in training mode
        self.model.train()

        # Return the calculated validation loss and validation accuracy
        return valid_loss, valid_accur

# Import data, all the data is in the (N, C, H, W) format (N - data samles, C - channels, H - height, W - width)
mnist_dir = '~/data/datasets/mnist' # Specify which directory to download MNIST to
# Get the data
train_data = torchvision.datasets.MNIST(root = mnist_dir, train = True, download = True, transform = transforms.ToTensor())
validation_data  = torchvision.datasets.MNIST(root = mnist_dir, train = False, transform = transforms.ToTensor())
# Split the data into inputs and outputs for training and desting
x_train, y_train = train_data.data, train_data.targets
x_valid,  y_valid  = validation_data.data,  validation_data.targets

# Reshape the input data to have 1 channel
x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
x_valid = x_valid.view(x_valid.shape[0], 1, x_valid.shape[1], x_valid.shape[2])

# Normalize the datasets to be between 0 and 1
x_train = x_train.float() / 255
x_valid  = x_valid.float()  / 255

# Set hyperparameters
BATCH_SIZE = 1000   # Batch size while training
N_EPOCHS   = 2      # Number of epochs for training

# Initialize the model
basic_MNIST_classifier = NetBasic()                                     # Set model to have NetBasic() architecture
loss_basic_model = nn.CrossEntropyLoss()                                # Set loss function to cross entropy loss
optimizer_basic_model = optim.Adam(basic_MNIST_classifier.parameters()) # Set optimizer to Adam

# Initialize the compiled model
compiled_basic_MNIST_classifier = CompiledModel(model = basic_MNIST_classifier, optimizer = optimizer_basic_model, loss_func = loss_basic_model)

# Train and tesst
if __name__ == "__main__":
    compiled_basic_MNIST_classifier.train(x_train, y_train, BATCH_SIZE, N_EPOCHS)
    calc_loss, calc_accur = compiled_basic_MNIST_classifier.validate(x_valid, y_valid)
    print(f'Final loss: {calc_loss}, Final accuracy: {calc_accur}')
