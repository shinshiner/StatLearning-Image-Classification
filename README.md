# StatLearning-Image-Classification'

The project of statistical learning (X033524).

## Requirements

* Python3.x
* pytorch
* sklearn

You can also check the detailed dependency in [requirements.txt](https://github.com/shinshiner/StatLearning-Image-Classification/blob/master/requirements.txt), and install them with [pip](https://pypi.org/project/pip/):

`
 pip3 install -r requirements.txt
`

## Usage

* Change your current dictionary into the root dictionary of this project.

* Run the command:

`python3 main --method [nn, sknn, svm or knn] --mode [train or test]`

## Tips

* In the training process, besides the best model and final model, I also save the model after several epoch. When you are testing, you should specific the model you want to use if you use `nn` method (default model is the final one).
