# Grad-CAM integration

### Sources
* Original paper: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
* Grad-CAM++ paper: [https://arxiv.org/pdf/1710.11063.pdf](https://arxiv.org/pdf/1710.11063.pdf)
* Grad-CAM++ original implementation: [https://github.com/adityac94/Grad_CAM_plus_plus](https://github.com/adityac94/Grad_CAM_plus_plus)

## Usage
### Preparing the network
* The 'nets' parameter of the network should contain the convolutional layers that can be relevant for Grad-CAM
### Configuration
* gradcam_record: True / False
* gradcam_layers: number of the relevant layers, in backward order (1 means only the last convolutional layer will be inspected)
* gradcam_layers_max: the number of the convolutional layers in the network (used only for naming)
* long_summary should be enabled at training

## Example
* network: ConvNet_mnist
* configuration: config_ConvNet_mnist