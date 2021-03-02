#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <assert.h>
#include <string.h>

#include "ours.h"

// Return a string of joined vector items.
template<typename T>
std::string join(const std::vector<T>& v)
{
    std::stringstream ss;
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            ss << ",";
        ss << v[i];
    }
    return ss.str();
}


int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void mnist_reader(std::string filepath, std::vector<std::vector<double>*>& data) {
    std::ifstream file (filepath);
    if (!file.good()) {
        std::cout << "file " << filepath << " failed to open" << std::endl;
        exit(-1);
    }

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            std::vector<double>* point = new std::vector<double>(n_rows * n_cols);
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    point->operator[](r * n_rows + c) = static_cast<double>(temp);
                }
            }
            data.push_back(point);
        }
    }
}


int main(int argc, char** argv) {
    assert(argc == 3 && " PASS IN \"MNIST\" OR \"SYNTHETIC\" AS ARGUMENT AS WELL AS NUMBER OF CENTERS (e.g ./a.out MNIST 2 --or-- ./a.out SYNTHETIC 2)");
    
    PrivateKMeans pkm;

    Result* r;

    int k = atoi(argv[2]);
    std::cout << "------------------------------------------------------ " << k << " CENTERS ------------------------------------------------------" << std::endl;
    if (strcmp(argv[1], "SYNTHETIC") == 0) {
        pkm.FULL_RANGE = 100*sqrt(100);

        std::mt19937 gen(500);
        int d = 100;
        int n = 50000;
        int data_k = 64;
        std::normal_distribution<double> ndist(0, 1);
        std::vector<std::vector<double>*> centers(data_k);
        for (int i = 0; i < data_k; i++) {
            centers[i] = new std::vector<double>(d);
            for (int s = 0; s < d; s++) {
                centers[i]->operator[](s) = 25 * ndist(gen);
            }
            // std::cout << "CENTER " << i << ": " << join(*centers[i]) << std::endl;
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
                if ((*npoint)[s] > 100) {
                    (*npoint)[s] = 100;
                } else if ((*npoint)[s] < 0) {
                    (*npoint)[s] = 0;
                }
            }
            theory_loss = pkm.l2_dist(*npoint, *centers[assigned_center]);
            data[i] = npoint;
        }

         // Write dataset to file
//         std::ofstream wfile("synth_dataset.txt");
//         wfile << d << " " << n << std::endl;
//         for (int i = 0; i < data.size(); i++) {
//             for (int dim = 0; dim < d; dim++) {
//                 wfile << data[i]->at(dim) << " ";
//             }
//             wfile << std::endl;
//         }
//         wfile.close();

        std::cout << "begin clustering synth" << std::endl;
        r = pkm.clustering(data, 1, k);
        //br = bkm.clustering(data_m, k, 1, 1);
        std::cout << std::endl << "---------------------------OUR LOSS: " << r->loss << std::endl;
        delete r;
        // std::cout << std::endl << "---------------------------BALCAN LOSS: " << br->loss << std::endl;
        // delete br;
        // std::cout << std::endl << "---------------------------THEORY LOSS: " << theory_loss << std::endl;

        // std::cout << std::endl << "balcan centers:" << std::endl << br->rec_centers << std::endl;

//        std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>* lloydsrun = pkm.lloydsalgo(data, k, 10);
//        double lloydloss = 0;
//        for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = lloydsrun->begin(); it != lloydsrun->end(); it++) {
//            for (int i = 0; i < it->second->size(); i++) {
//                lloydloss += pkm.l2_dist(*it->first, *it->second->at(i));
//            }
//            delete it->first;
//            delete it->second;
//        }
//        delete lloydsrun;
//        std::cout << "---------------------------LLOYDS LOSS: " << lloydloss << std::endl;

        for (int i = 0; i < k; i++) {
            delete centers[i];
        }

        for (int i = 0; i < n; i++) {
            delete data[i];
        }
    }
    else if (strcmp(argv[1], "MNIST") == 0) {
        std::cout << argv[1] << std::endl;
        pkm.FULL_RANGE = 255*sqrt(784);

//        std::string filepath("train-images.idx3-ubyte");
//        std::ifstream file (filepath);
//        if (!file.good()) {
//            std::cout << "file " << filepath << " failed to open" << std::endl;
//            exit(-1);
//        }
        std::vector<std::vector<double>*> data;
        mnist_reader("train-images.idx3-ubyte", data);
        // mnist_reader("t10k-images.idx3-ubyte", data);
//        if (file.is_open())
//        {
//            int magic_number=0;
//            int number_of_images=0;
//            int n_rows=0;
//            int n_cols=0;
//            file.read((char*)&magic_number,sizeof(magic_number));
//            magic_number= reverseInt(magic_number);
//            file.read((char*)&number_of_images,sizeof(number_of_images));
//            number_of_images= reverseInt(number_of_images);
//            file.read((char*)&n_rows,sizeof(n_rows));
//            n_rows= reverseInt(n_rows);
//            file.read((char*)&n_cols,sizeof(n_cols));
//            n_cols= reverseInt(n_cols);
//            for(int i=0;i<number_of_images;++i)
//            {
//                std::vector<double>* point = new std::vector<double>(n_rows * n_cols);
//                for(int r=0;r<n_rows;++r)
//                {
//                    for(int c=0;c<n_cols;++c)
//                    {
//                        unsigned char temp=0;
//                        file.read((char*)&temp,sizeof(temp));
//                        point->operator[](r * n_rows + c) = static_cast<double>(temp);
//                    }
//                }
//                data.push_back(point);
//            }
//        }
        // int d = data[0]->size();
        // int n = data.size();
        // Eigen::MatrixXd m_data(d, n);
        // for (int i = 0; i < n; i++) {
        //     for (int s = 0; s < d; s++) {
        //         m_data(s, i) = data[i]->at(s);
        //     }
        // }

        r = pkm.clustering(data, 1, k);
        // br = bkm.clustering(m_data, k, 1, 1);
        std::cout << std::endl << "---------------------------OUR LOSS: " << r->loss << std::endl;
        delete r;
        // std::cout << std::endl << "---------------------------BALCAN LOSS: " << br->loss << std::endl;
        // delete br;
        std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>* lloydsrun = pkm.lloydsalgo(data, k, 10);
        double lloydloss = 0;
        for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = lloydsrun->begin(); it != lloydsrun->end(); it++) {
            for (int i = 0; i < it->second->size(); i++) {
                lloydloss += pkm.l2_dist(*it->first, *it->second->at(i));
            }
            delete it->first;
            delete it->second;
        }
        delete lloydsrun;
        std::cout << "---------------------------LLOYDS LOSS: " << lloydloss << std::endl;


        for (int i = 0; i < data.size(); i++) {
            delete data[i];
        }
    } else {
        std::vector<std::vector<double>*> data;
        std::ifstream infile("USDATACOMPESSEDFIXED.txt");
        std::string delim = ",";
        if (infile.is_open()) {
            std::string s;
            int points = 0;
            while (std::getline(infile, s)) {
                if (points >= 49999) {
                    break;
                }
                std::vector<double>* holder = new std::vector<double>(68);
                auto start = 0U;
                auto end = s.find(delim);
                int i = 0;
                while (end != std::string::npos)
                {
                    if (i != 0) {
                        holder->operator[](i - 1) = std::stod(s.substr(start, end - start));
                    }
                    start = end + delim.length();
                    end = s.find(delim, start);
                    i++;
                }

                holder->operator[](i - 1) = std::stod(s.substr(start, end));
                data.push_back(holder);
                std::getline(infile, s);
                points++;
            }
        }
        infile.close();

        PrivateKMeans pkm;
        pkm.FULL_RANGE = sqrt(68) * 223;
        Result* r = pkm.clustering(data, 1, k);
        std::cout << std::endl << "---------------------------OUR LOSS: " << r->loss << std::endl;
        delete r;
        std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>* lloydsrun = pkm.lloydsalgo(data, k, 10);
        double lloydloss = 0;
        for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = lloydsrun->begin(); it != lloydsrun->end(); it++) {
            for (int i = 0; i < it->second->size(); i++) {
                lloydloss += pkm.l2_dist(*it->first, *it->second->at(i));
            }
            delete it->first;
            delete it->second;
        }
        delete lloydsrun;
        std::cout << "---------------------------LLOYDS LOSS: " << lloydloss << std::endl;

        for (int i = 0; i < data.size(); i++) {
            delete data[i];
        }
    }
}