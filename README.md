## Overview

Code and implementation for our differentially private k-means submission. The implementation is contained entirely inside the header file named `ours.h`. It does not rely upon any external libraries and is entirely self-contained.

To conduct an experiment execute the following code.
```
g++ -std=c++17 our_reader.cpp
./a.out MNIST 2
```

The last command runs the algorithm on the MNIST dataset with 2 centres. In order to run the algorithm on the synthetic dataset used in the paper, simply run `./a.out SYNTHETIC 2` instead. To change the number of centres, simply change the last number to the desired `k`. 

`synth.txt` and `test.txt` is the data used to generate the graphs in the paper for the synthetic and MNIST datasets, respectively.

The full implementation of the algorithm resides in `ours.h`.

## MNIST Dataset

Before running the MNIST dataset, it is first necessary to place the dataset file at the root of this repo. The MNIST dataset can be found at the following link: http://yann.lecun.com/exdb/mnist/. Only the `train-images-idx3-ubyte.gz` file is needed from this website. Place the downloaded file as is at the root of this repo (in the same directory as `ours.cpp`), and simply run the command from above (`./a.out MNIST 2`) in order to run an experiment on the MNIST dataset, where `a.out` is of course the binary executable file that results from compiling `our_reader.cpp`.

## Synthetic Dataset

Nothing needs to be done for the synthetic dataset as the dataset used in the paper is generated at run time using the same seed.

## How to use
In order to run our code, first create a dataset of type `std::vector<std::vector<double>*>` where each element is a datapoint. Next, construct an instance of the class PrivateKMeans. Before you call any method, set the field `FULL_RANGE` within the class to specify the diameter of the dataset. Once the field `FULL_RANGE` has been specified and instantiated with the appropriate value for your dataset, call the method `clustering(...)` which takes three parameters: the dataset in the form of a vector of vector points (the aforementioned type), the privacy parameter, and the number of centres. The function will return a pointer to a class called `Results` which simply holds two fields; the loss of the algorithm and an `unordered_map` mapping a centre to the list of datapoints assigned to it. Below is some sample code illustrating the instructions above.

```C++
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <assert.h>
#include <string.h>

#include "ours.h"

int main(int argc, char** argv) {
    // Instantiating class and necessary field FULL_RANGE
    PrivateKMeans pkm;
    pkm.FULL_RANGE = 100*sqrt(100);
    
    // Constructing dataset
    std::mt19937 gen(500);
    int d = 100;
    int k = 16;
    int n = 50000;
    int data_k = 64;
    std::normal_distribution<double> ndist(0, 1);
    std::vector<std::vector<double>*> centers(data_k);
    for (int i = 0; i < data_k; i++) {
        centers[i] = new std::vector<double>(d);
        for (int s = 0; s < d; s++) {
            centers[i]->operator[](s) = 100 * ndist(gen);
        }
    }
    std::cout << std::endl;

    int n0 = n / data_k;

    double theory_loss = 0;
    std::vector<std::vector<double>*> data(n);
    for (int i = 0; i < n; i++) {
        std::vector<double>* npoint = new std::vector<double>(d);
        int assigned_center = i / n0;
        if (assigned_center >= data_k) {
            assigned_center = data_k - 1;
        }
        for (int s = 0; s < d; s++) {
            (*npoint)[s] = ndist(gen) + centers[assigned_center]->at(s);
        }
        theory_loss = pkm.l2_dist(*npoint, *centers[assigned_center]);
        data[i] = npoint;
    }

    // Running the clustering algorithm with k centers, privacy parameter 1, and on the dataset data.
    std::cout << "begin clustering synth" << std::endl;
    r = pkm.clustering(data, 1, k);
    std::cout << std::endl << "---------------------------OUR LOSS: " << r->loss << std::endl;
    delete r;

    for (int i = 0; i < k; i++) {
        delete centers[i];
    }

    for (int i = 0; i < n; i++) {
        delete data[i];
    }
}
```
