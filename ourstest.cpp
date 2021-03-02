#include <assert.h>
#include <iostream>
#include <sstream>

#include "ours.h"
#include <list>
#include <vector>
#include <unordered_map>

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

void test_combination_generation() {
    std::cout << "TEST COMBINATION_GENERATION" << std::endl;
    PrivateKMeans pkm;
    std::vector<std::vector<int>*>* partitions = pkm.square_sum_generator(5, 5);
    // for (int i = 0; i < partitions->size(); i++) {
    //     for (int j = 0; j < partitions->at(i)->size(); j++) {
    //         std::cout << partitions->at(i)->at(j) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << partitions->size() << std::endl;
    for (int i = 0; i < partitions->size(); i++) {
        delete partitions->at(i);
    }
    delete partitions;
    
}

// void test_capture_centers() {
//     std::cout << "TEST CAPTURE CENTERS" << std::endl;
//     PrivateKMeans pkm;
//     std::unordered_map<std::string, int> assignments;
//     std::vector<std::vector<int>*> reprv;
//     std::vector<int> acc{0, 0};
//     std::vector<int> offsets{1, 2};
//     std::vector<double> point{-1.3, 2.5};
//     std::unordered_map<std::string, std::vector<std::vector<double>*>*> tass;
//     pkm.capture_centers(point, offsets, assignments, 0, acc, 10, .5, tass, reprv);
//     int i = 0;
//     for (std::unordered_map<std::string, int>::iterator it = assignments.begin(); it != assignments.end(); it++) {
//         std::cout << it->first << std::endl;
//         std::cout << std::endl;
//         delete reprv[i];
//         delete tass[it->first];
//         i++;
//     }
// }

// void test_get_candidate_centers() {
//     std::cout << "TEST CANDIDATE CENTERS" << std::endl;
//     PrivateKMeans pkm;
//     std::vector<double> p1 = {0.25, 0.25, 0.25};
//     std::vector<double> p2 = {0.5, 0.5, 0.5};
//     std::vector<double> p3 = {0.75, 0.75, 0.8};
//     std::cout << &p1 << std::endl << &p2 << std::endl << &p3 << std::endl;
//     List nl;
//     nl.push_back(&p1);
//     nl.push_back(&p2);
//     nl.push_back(&p3);
//     std::vector<std::vector<double>*> reservoir;
//     double r = 0.25;
//     std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*> assignments;
//     pkm.get_candidate_centers(nl, 2, 3, 1, reservoir, assignments, r);


//     std::cout << std::endl;
//     for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = assignments.begin(); it != assignments.end(); it++) {
//         std::cout << join(*it->first) << std::endl;
//         for (int i = 0; i < it->second->size(); i++) {
//             std::cout << "     - " << join(*it->second->at(i)) << std::endl;
//         }
//     }
//     for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = assignments.begin(); it != assignments.end(); it++) {
//         delete it->second;
//     }
//     for (int i = 0; i < reservoir.size(); i++) {
//         delete reservoir[i];
//     }
// }

// void test_transform() {
//     std::cout << "TEST TRANSFORM" << std::endl;
//     PrivateKMeans pkm(5);
//     std::vector<double> p1 = {3, -3, 3};
//     std::vector<double> p2 = {5, 5, 5};
//     std::vector<double> p3 = {10, -11, 13};
//     std::vector<std::vector<double>*> data = {&p1, &p2, &p3};
//     std::vector<std::vector<double>*>* vals = pkm.transform_data(data, 2);
//     for (int i = 0; i < vals->size(); i++) {
//         std::cout << join(*vals->at(i)) << std::endl;
//     }
//     for (int i = 0; i < vals->size(); i++) {
//         delete vals->at(i);
//     }
//     delete vals;
// }

// void test_lloyd() {
//     std::cout << "TEST LLOYD" << std::endl;
//     PrivateKMeans pkm(5);
//     std::vector<double> p1 = {10, -10};
//     std::vector<double> p2 = {10, 10};
//     std::vector<double> p3 = {-3, 2};
//     std::vector<double> p4 = {-10, -10};
//     std::vector<std::vector<double>*> data = {&p1, &p2, &p3, &p4};
//     std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>& assign = *(pkm.lloydsalgo(data, 3, -1));

//     for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = assign.begin(); it != assign.end(); it++) {
//         std::cout << "CENTER: " << join(*it->first) << std::endl;
//         for (int i = 0; i < it->second->size(); i++) {
//             std::cout << "     - " << join(*it->second->at(i)) << std::endl;
//         }
//         delete it->second;
//         delete it->first;
//     }
//     delete &assign;
// }

// void test_pkm() {
//     std::cout << "TEST PRIVATE K MEANS" << std::endl;
//     PrivateKMeans pkm;
//     pkm.FULL_RANGE = sqrt(2) * 10;
//     std::vector<double> p1 = {10, -100};
//     std::vector<double> p2 = {10, 100};
//     std::vector<double> p3 = {-10, 100};
//     std::vector<double> p4 = {-10, -100};
//     std::vector<std::vector<double>*> data = {&p1, &p2, &p3, &p4};
//     Result* ret = pkm.clustering(data, 1, 1, 1, 4);
//     Result& r = *ret;
//     std::cout << "LOSS: " << r.loss << std::endl;
//     for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = r.clusters.begin(); it != r.clusters.end(); it++) {
//         std::cout << join(*it->first) << std::endl;
//         for (int i = 0; i < it->second->size(); i++) {
//             std::cout << "     - " << join(*it->second->at(i)) << std::endl;
//         }
//     }
//     delete ret;
// }

class tempo {
    public:
        int val;
};


int main(int argc, char** argv) {
    //test_transform();
    //test_lloyd();
    // test_pkm();
    //test_get_candidate_centers();
    test_combination_generation();
    size_t temp = 5;
    int s = 6;
    std::cout << temp - s << std::endl;
}