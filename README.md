# Depth-wise Separable Convolution Neural Network with Residual Connection for Hyperspectral Image Classification
The Code for "Depth-wise Separable Convolution Neural Network with Residual Connection for Hyperspectral Image Classification"

If you use this code, please cite our work:  https://doi.org/10.3390/rs12203408

Dang, L.; Pang, P.; Lee, J. Depth-Wise Separable Convolution Neural Network with Residual Connection for Hyperspectral Image Classification. Remote Sens. 2020, 12, 3408. doi: 10.3390/rs12203408


## Requirements


> matplotlib             3.1.1
>
> python                 3.6.5
>
> numpy                  1.17.2+mkl
>
> opencv-python          4.1.1.26
>
> scikit-learn           0.22.1
>
> sklearn                0.0
>
> spectral               0.19
>
> torch                  1.0.0

## Data sets
You can download the hyperspectral data sets in matlab format at:
http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

## Example of use

* Execute the "main.py" file to complete the training and testing

```shell script
python main.py
```


* Illustrate final classification maps:

```shell script
python show_pred_map.py
```
