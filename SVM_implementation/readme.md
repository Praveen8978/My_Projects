
# SVM Implementation using PyTorch on Iris Dataset

This repository contains an implementation of a Support Vector Machine (SVM) using PyTorch, trained on the Iris dataset. The goal of the project is to classify the Iris dataset using a linear SVM and to visualize the results of the training process.

## Requirements

- Python 3.9
- PyTorch
- Matplotlib
- NumPy
- Scikit-learn

You can install the necessary dependencies using `pip`:

```bash
pip install torch matplotlib numpy scikit-learn
```

## Dataset

The dataset used is the **Iris dataset**, which is widely used for classification problems. It consists of 150 samples with 4 features each (sepal length, sepal width, petal length, and petal width) and three possible classes (setosa, versicolor, virginica). However, for the purposes of this SVM implementation, the dataset is transformed into a binary classification problem by assigning:

- `+1` for **setosa**
- `-1` for **other species (versicolor and virginica)**

This allows us to focus on a binary classification task.

## References

- [SVM with PyTorch - BytePawn](https://bytepawn.com/svm-with-pytorch.html)
- [PyTorch Tutorial by Patrick Loeber](https://github.com/patrickloeber/pytorchTutorial/tree/master)

## Improvements

Currently this is a binary classifier. It can be made multi class classifier by implementing 
- one vs one (or)
- one vs all