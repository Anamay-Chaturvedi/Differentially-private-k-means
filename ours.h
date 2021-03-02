#pragma once

#include <vector>
#include <assert.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <sstream>
#include <math.h>
#include <list>
#include <random>
#include <experimental/algorithm>
#include <algorithm>



class NodeList {
    public:
        NodeList* next = nullptr;
        NodeList* prev = nullptr;
        std::vector<double>* point;

        NodeList(std::vector<double>* point) {
            this->point = point;
        }

        void add(std::vector<double>* point) {
            NodeList* nl = new NodeList(point);
            nl->next = this;
            prev = nl;
        }

        NodeList* remove() {
            if (prev != nullptr) {
                prev->next = next;
            }
            if (next != nullptr) {
                next->prev = prev;
            }
            return this;
        }

        void add(NodeList* nl) {
            prev = nl;
            nl->next = this;
        }
};

class List {
    public:
        NodeList* begin = nullptr;
        NodeList* removed = nullptr;
        NodeList* first_delete = nullptr;
        std::unordered_map<NodeList*, int> delv;
        size_t sizei = 0;
        size_t removedi = 0;

        void push_back(std::vector<double>* point) {
            sizei++;
            if (begin == nullptr) {
                begin = new NodeList(point);
                return;
            }
            begin->add(point);
            begin = begin->prev;
        }

        bool remove(NodeList* nv) {
            if (delv.find(nv) == delv.end()) {
                if (nv == begin) {
                    begin = begin->next;
                }
                delv[nv->remove()] = -1;
                sizei--;
                removedi++;
                nv->next = nullptr;
                nv->prev = nullptr;
                
                if (removed == nullptr) {
                    removed = nv;
                    first_delete = nv;
                } else {
                    removed->add(nv);
                    removed = nv;
                }
                return true;
            }
            return false;
        }
        
        ~List() {
            for (std::unordered_map<NodeList*, int>::iterator it = delv.begin(); it != delv.end(); it++) {
                delete it->first;
            }
            if (begin != nullptr) {
                NodeList* iter = begin;
                NodeList* next = begin->next;
                while (iter != nullptr) {
                    delete iter;
                    iter = next;
                    if (next != nullptr) {
                        next = next->next;
                    }
                }
            }
        }

        void reset() {
            if (first_delete == nullptr) {
                return;
            }
            delv.clear();
            sizei += removedi;
            removedi = 0;
            if (begin == nullptr) {
                begin = removed;
                removed = nullptr;
                first_delete = nullptr;
                return;
            }
            first_delete->next = begin;
            begin->prev = first_delete;
            begin = removed;
            removed = nullptr;
            first_delete = nullptr;
        }

        bool deleted(NodeList* n) {
            return delv.find(n) != delv.end();
        }

        size_t size() {
            return sizei;
        }
};



class Result {

    public:
        std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*> clusters;
        double loss = 0;

        ~Result() {
            for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = clusters.begin(); it != clusters.end(); it++) {
                delete it->second;
                delete it->first;
            }
        }
};


class PrivateKMeans {

    private:


        /**
         * @brief Helper method for generating all possible combinations.
         * 
         * @param ref_array Array holding all positions of the partitions.
         * @param index The current partition we are considering.
         * @param reserve Where to store the combinations.
         * @param goal_sum The sum we are trying to achieve.
         */
        void balls_in_bins_helper(std::vector<int>& ref_array, int index, 
            std::vector<std::vector<int>*>* reserve, int goal_sum) {
            
            if (index == ref_array.size() - 1) {
                for (int i = 0; i <= static_cast<int>(sqrt(goal_sum)); i++) {
                    ref_array[index] = i;
                    std::vector<int>* temp = new std::vector<int>(ref_array);
                    reserve->push_back(temp);
                }
                return;
            }

            for (int i = 0; i <= static_cast<int>(sqrt(goal_sum)); i++) {
                int new_goal = goal_sum - static_cast<int>(pow(i, 2));
                ref_array[index] = i;
                if (new_goal < 0) {
                    break;
                }
                balls_in_bins_helper(ref_array, index + 1, reserve, new_goal);
            }
            
        }

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

    public:
        double FULL_RANGE = -1;
        double min_side = 0;
        double max_side = 100;
        
        std::random_device rd;
        std::mt19937 gen;

        PrivateKMeans() {
            int seed = rd();
            std::cout << "      - SEED: " << seed << std::endl;
            gen.seed(seed);
        }

        PrivateKMeans(int seed) {
             gen.seed(seed);
        }

        /**
         * @brief Generates all possible combinations for bin dimensional integer valued vectors
         * such that the square of all elements is less than or equal to the goal sum.
         * 
         * @param bins The number of bins.
         * @param balls The number of balls.
         * @return std::vector<std::vector<int>*>* 
         */
        std::vector<std::vector<int>*>* square_sum_generator(int bins, int goal_sum) {
            assert(bins > 0 && goal_sum >= 0);
            std::vector<std::vector<int>*>* newarray = new std::vector<std::vector<int>*>();
            std::vector<int> refarray(bins);
            balls_in_bins_helper(refarray, 0, newarray, goal_sum);
            return newarray;
        }

        /**
         * @brief Samples the given vector.
         * 
         * @param v The vector with probability values.
         * @param full_size The sum of the vector given.
         * @return int The index chosen.
         */
        int sample(const std::vector<double>& v, double full_size) {
            std::uniform_real_distribution<double> unifr(0, full_size);
            double count = unifr(gen);
            double curr_count = 0;
            for (int i = 0; i < v.size(); i++) {
                curr_count += v[i];
                if (curr_count >= count && v[i] != 0) {
                    return i;
                }
            }
            return v.size() - 1;
        }

        /**
         * @brief Get the candidate centers object
         * 
         * @param m_data The data points
         * @param k The number of centers to choose.
         * @param d The dimension
         * @param epsilon The privacy epsilon parameter.
         * @param reservoir Where to store the chosen candidate centers.
         * @param chosen_assignments Map of candidate centers to the index of points assigned to them. 
         * @param r The radius.
         */
        void get_candidate_centers(std::vector<std::vector<double>*>& m_data, int k, int d, double epsilon, std::vector<std::vector<double>*>& reservoir, std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>& chosen_assignments, double r) {
            if (m_data.size() == 0) {
                return;
            }
            std::vector<std::vector<int>*>* offset_mat = square_sum_generator(d, static_cast<int>(d / pow(epsilon, 2) * pow(1 + epsilon, 2)));
            
            std::cout << "    Finding Candidate Centers with radius " << r << std::endl;
            for (int round = 0; round < k; round++) {
                if (m_data.size() == 0) {
                    for (int i = 0; i < offset_mat->size(); i++) {
                        delete offset_mat->at(i);
                    }
                    delete offset_mat;
                    return;
                }
                std::unordered_map<std::string, int> tass;
                std::vector<int> acc(d);
                std::vector<std::vector<int>*> reprv;
                std::unordered_map<std::string, std::vector<std::vector<double>*>*> assignments;
                for (int i = 0; i < m_data.size(); i++) {
                    for (int j = 0; j < offset_mat->size(); j++) {
                        capture_centers(*m_data[i], *offset_mat->at(j), tass, 0, acc, static_cast<int>(ceil(1 / r)), r, assignments, reprv);
                    }
                    tass.clear();
                }
                std::cout << "REPRV SIZE: " << reprv.size() << std::endl;
                int total_size = 0;
                std::vector<double> probs(reprv.size());
                for (std::unordered_map<std::string, std::vector<std::vector<double>*>*>::iterator it = assignments.begin(); it != assignments.end(); it++) {
                    total_size += it->second->size();
                }

                std::bernoulli_distribution bdist(total_size * 1.0 / (total_size * 1.0 + (1 / pow(r, d) + reprv.size())));

                double prob_size = 0;
                double tval;
                int ind_max = 0;
                size_t curr_max = 0;
                for (int i = 0; i < reprv.size(); i++) {
                    
                    std::string vcs = join(*reprv[i]);
                    tval = exp(epsilon * assignments[vcs]->size() / 2.0);
                    probs[i] = tval;
                    prob_size += tval;
                    // std::cout <<  assignments[vcs]->size() << std::endl;
                    if (assignments[vcs]->size() > curr_max) {
                        ind_max = i;
                        curr_max = assignments[vcs]->size();
                    }
                }

                int ind;
                if (bdist(gen)) {
                    ind = sample(probs, prob_size);
                } else {
                    ind = std::uniform_int_distribution<int>{0, static_cast<int>(reprv.size() - 1)}(gen);
                }
                ind = ind_max;
                
                std::vector<int>* cand_center_to_add = reprv[ind];
                std::vector<double>* n_center = new std::vector<double>();
                for (int i = 0; i < cand_center_to_add->size(); i++) {
                    n_center->push_back(cand_center_to_add->at(i) * r);
                }
                reservoir.push_back(n_center);
                chosen_assignments[n_center] = assignments[join(*cand_center_to_add)];

                std::vector<std::vector<double>*>* chosen_v = assignments[join(*cand_center_to_add)];
                for (int i = 0; i < chosen_v->size(); i++) {
                    std::vector<std::vector<double>*>::iterator din = std::find(m_data.begin(), m_data.end(), chosen_v->at(i));
                    if (din == m_data.end()) {
                        std::cout << "NOT FOUND" << join(*chosen_v->at(i)) << std::endl;
                        continue;
                    }
                    m_data.erase(din);
                }
                for (int i = 0; i < reprv.size(); i++) {
                    if (reprv[i] == cand_center_to_add) {
                        delete reprv[i];
                        continue;
                    }
                    delete assignments[join(*reprv[i])];
                    delete reprv[i];
                }
            }
            for (int i = 0; i < offset_mat->size(); i++) {
                delete offset_mat->at(i);
            }
            delete offset_mat;
        }




        /**
         * @brief Given a point, finds all grid points that are only an offset away from the given point.
         * 
         * @param point The point.
         * @param offsets The offset to consider.
         * @param assignments Map to check if we have already encountered a point.
         * @param index The current index we are considering.
         * @param reprv The vector holding all new grid points.
         * @param acc The grid point we are currently building.
         * @param limit The maximium grid point a point may take.
         * @param r The current side length of the net.
         * @param tassignments Global assignments.
         */
        void capture_centers(std::vector<double>& point, std::vector<int>& offsets, std::unordered_map<std::string, int>& assignments, int index, std::vector<int>& acc, int limit, double r, std::unordered_map<std::string, std::vector<std::vector<double>*>*>& tassignments, std::vector<std::vector<int>*>& treprv) {
            if (index == point.size() - 1) {
                std::vector<int>* temp;
                for (int j = 0; j < 2; j++) {
                    offsets[index] *= -1;
                    if (offsets[index] >= 0) {
                        if (offsets[index] > 0 || j == 0) {
                            acc[index] = static_cast<int>(ceil(point[index] / r)) + offsets[index];
                            if (acc[index] <= -limit) {
                                acc[index] = -limit;
                            } else if (acc[index] >= limit) {
                                acc[index] = limit;
                            }
                        }
                    }
                    if (offsets[index] <= 0) {
                        if (offsets[index] < 0 || j == 1) {
                            acc[index] = static_cast<int>(floor(point[index] / r)) + offsets[index];
                            if (acc[index] <= -limit) {
                                acc[index] = -limit;
                            } else if (acc[index] >= limit) {
                                acc[index] = limit;
                            }
                        }
                    }
                    std::string vc = join(acc);
                    if (assignments.find(vc) == assignments.end()) {
                        assignments[vc] = -1;
                        if (tassignments.find(vc) == tassignments.end()) {
                            temp = new std::vector<int>(acc);
                            treprv.push_back(temp);
                            tassignments[vc] = new std::vector<std::vector<double>*>();
                        }
                        tassignments[vc]->push_back(&point);
                    }
                }
                return;
            }
            
            
            offsets[index] *= -1;
            if (offsets[index] <= 0) {
                acc[index] = static_cast<int>(floor(point[index] / r)) + offsets[index];
                if (acc[index] <= -limit) {
                    acc[index] = -limit;
                } else if (acc[index] >= limit) {
                    acc[index] = limit;
                }
                capture_centers(point, offsets, assignments, index + 1, acc, limit, r, tassignments, treprv);
            }
            offsets[index] *= -1;
            if (offsets[index] >= 0) {
                acc[index] = static_cast<int>(ceil(point[index] / r)) + offsets[index];
                if (acc[index] <= -limit) {
                    acc[index] = -limit;
                } else if (acc[index] >= limit) {
                    acc[index] = limit;
                }
                capture_centers(point, offsets, assignments, index + 1, acc, limit, r, tassignments, treprv);
            }
        }

        Result* clustering(const std::vector<std::vector<double>*>& data, double epsilon1, int k) {
            assert(FULL_RANGE != -1 && "FULL_RANGE NOT SET: SET THE FIELD FULL_RANGE BEFORE RUNNING");
            int old_dim = data[0]->size();
            int new_dim = static_cast<int>(log(data.size()) / 2);
            std::cout << "NEW_DIM: " << new_dim << std::endl;
            if (data[0]->size() <= 5) {
                new_dim = data[0]->size();
            }
            std::vector<std::vector<double>*>* transformed_data = transform_data(data, new_dim);
            std::unordered_map<std::vector<double>*, int> rev_indices;
            List ll;
            //std::vector<std::vector<double>*> lllist(data.size());
            for (int i = 0; i < transformed_data->size(); i++) {
                rev_indices[transformed_data->at(i)] = i;
                ll.push_back(transformed_data->at(i));
                //lllist[i] = transformed_data->at(i);
            }
            double r = epsilon1 / (data.size() * sqrt(new_dim)) * 1000;
            double rad = 1.0 / (data.size()) * 1000;
            std::vector<std::vector<double>*> reservoir;
            std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*> chosen_assignments;
            while (r <= 1) {
                get_candidate_centers(ll, static_cast<int>(k / epsilon1), new_dim, (4 / (exp(1.0) * 3 * log(2 * data.size()))) * (epsilon1 / 10), reservoir, chosen_assignments, r, rad);
                // for (int i = 0; i < reservoir.size(); i++) {
                //     std::cout << join(*reservoir[i]) << std::endl;
                // }
                // ll.reset();
                r = 2 * r;
                rad = 2 * rad;
            }

            // Assigning points to nearest candidate centers
            std::cout << "    ASSEMBLING D\'\'" << std::endl;
            std::exponential_distribution<double> expdist(1 / (1 / (epsilon1 / 10)));
            std::bernoulli_distribution berdist(0.5);
            std::unordered_map<std::vector<double>*, int> weights;
            for (int i = 0; i < transformed_data->size(); i++) {
                double min_dist = 10000000000;
                int ind = -1;
                for (int j = 0; j < reservoir.size(); j++) {
                    double tempdist = sqrt(l2_dist(*reservoir[j], *transformed_data->at(i)));
                    if (tempdist < min_dist) {
                        min_dist = tempdist;
                        ind = j;
                    }
                }
                weights[reservoir[ind]] += 1;
            }
            for (int i = 0; i < reservoir.size(); i++) {
                int to_add = static_cast<int>(expdist(gen) * (2 * berdist(gen) - 1));
                if (weights.find(reservoir[i]) == weights.end()) {
                    if (to_add > 0) {
                        weights[reservoir[i]] = to_add;
                    }
                    continue;
                }
                weights[reservoir[i]] += to_add;
                if (weights[reservoir[i]] <= 0) {
                    weights.erase(weights.find(reservoir[i]));
                }
            }

            std::cout << "    RUNNING LLOYDS ON D\'\', SIZE: " << weights.size() << std::endl;
            std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>& cand_clusters = *(wlloydsalgo(weights, k, -1));
            
            Result* resp = new Result();
            Result& res = *resp;
            std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*> finder;
            for (int i = 0; i < transformed_data->size(); i++) {
                double min_dist = -1;
                std::vector<double>* closest = nullptr;
                for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = cand_clusters.begin(); it != cand_clusters.end(); it++) {
                    if (closest == nullptr) {
                        min_dist = sqrt(l2_dist(*it->first, *transformed_data->at(i)));
                        closest = it->first;
                        continue;
                    }
                    double temp_dist = sqrt(l2_dist(*transformed_data->at(i), *it->first));
                    if (temp_dist < min_dist) {
                        min_dist = temp_dist;
                        closest = it->first;
                    }
                }
                if (finder.find(closest) == finder.end()) {
                    finder[closest] = new std::vector<std::vector<double>*>();
                }
                finder[closest]->push_back(data[rev_indices[transformed_data->at(i)]]);
            }
            std::cout << "FINDER SIZE " << finder.size() << std::endl;
            for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = finder.begin(); it != finder.end(); it++) {
                std::vector<std::vector<double>*>* holder = it->second;
                std::vector<double>* nav = new std::vector<double>(old_dim);
                noisy_average(*holder, *nav, old_dim, epsilon1 * 10 / 25.0, 1 / (2*pow(data.size(), 1.5)), 'G');
                for (int lks = 0; lks < it->second->size(); lks++) {
                    res.loss += l2_dist(*nav, *it->second->at(lks));
                }
                res.clusters[nav] = holder;
            }
            std::cout << "  BEFORE LLOYDS LOSS: " << res.loss << std::endl;
            // for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = cand_clusters.begin(); it != cand_clusters.end(); it++) {
            //     std::vector<std::vector<double>*>* tarray = new std::vector<std::vector<double>*>();
            //     // std::cout << "CENTER " << " POINTS: ";
            //     for (int i = 0; i < it->second->size(); i++) {
            //         std::vector<std::vector<double>*>* ccpoints = chosen_assignments[it->second->at(i)];
            //         for (int j = 0; j < ccpoints->size(); j++) {
            //             tarray->push_back(data[rev_indices[ccpoints->at(j)]]);
            //             // std::cout << rev_indices[ccpoints->at(j)] << " ";
            //             // delete ccpoints->at(j);
            //         }
            //     }
            //     std::cout << tarray->size() << std::endl;
            //     std::vector<double>* nav = new std::vector<double>(old_dim);
            //     noisy_average(*tarray, *nav, data[0]->size(), epsilon1);
            //     std::cout << "CENTER " << ": " << join(*nav) << std::endl;
            //     res.clusters[nav] = tarray;
            //     for (int i = 0; i < tarray->size(); i++) {
            //         res.loss += l2_dist(*nav, *tarray->at(i));
            //     }
            //     //checker.clear();
            // }

            // Running 8 iterations of lloyd's for improvement in result
            int lloyditers = 1;
            std::cout << "BEGINNING " << lloyditers << " ROUNDS OF LLOYD " << data.size() << std::endl;
            double dist;
            std::vector<double>* closest;
            res.loss = 0;
            for (int i = 0; i < lloyditers; i++) {
                for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = res.clusters.begin(); it != res.clusters.end(); it++) {
                    it->second->clear();
                }
                std::cout << "cleared" << std::endl;
                for (int z = 0; z < data.size(); z++) {
                    dist = 1000000000;
                    for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = res.clusters.begin(); it != res.clusters.end(); it++) {
                        double comp = sqrt(l2_dist(*it->first, *data[z]));
                        if (comp < dist) {
                            dist = comp;
                            closest = it->first;
                        }
                    }
                    res.clusters[closest]->push_back(data[z]);
                }
                std::cout << "Reassigned" << std::endl;
                for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = res.clusters.begin(); it != res.clusters.end(); it++) {
                    noisy_average(*it->second, *it->first, data[0]->size(), epsilon1 * 10 / 25.0, 1 / (2*pow(data.size(), 1.5)), 'G');
                }
                if (i == lloyditers - 1) {
                    // for (int lks = 0; lks < it->second->size(); lks++) {
                    //     res.loss += l2_dist(*it->first, *it->second->at(lks));
                    // }
                    for (int st = 0; st < data.size(); st++) {
                        double min_dist = 100000000000000;
                        for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator itsec = res.clusters.begin(); itsec != res.clusters.end(); itsec++) {
                            double temp_dist = l2_dist(*itsec->first, *data[st]);
                            if (temp_dist < min_dist) {
                                min_dist = temp_dist;
                            }
                        }
                        res.loss += min_dist;
                    }
                }
                std::cout << "recovered" << std::endl;
            }

            for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = cand_clusters.begin(); it != cand_clusters.end(); it++) {
                delete it->second;
                delete it->first;
            }
            delete &cand_clusters;

            for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = chosen_assignments.begin(); it != chosen_assignments.end(); it++) {
                delete it->second;
                delete it->first;
            }

            for (int i = 0; i < transformed_data->size(); i++) {
                delete transformed_data->at(i);
            }
            delete transformed_data;
            return resp;
        }


        /**
         * @brief Computers a noisy average from the given list of points.
         * 
         * @param points 
         * @param epsilon 
         * @return std::vector<double>* 
         */
        void noisy_average(const std::vector<std::vector<double>*>& points, std::vector<double>& dest, int d, double epsilon, double delta, char dist_type) {
            assert(dest.size() == d);
            vaverage(points, dest);

            if (dist_type == 'N') {
                std::cout << "RETURNED IN NOISY AVERAGE" << std::endl;
                return;
            }

            double gauss_variation = 1;//FULL_RANGE * log(5/(4*delta)) / (points.size() * epsilon);
            std::normal_distribution<double> dist(0, 1);
            // std::cout << " REASON " << FULL_RANGE * sqrt(log(1/delta)) / (points.size() * epsilon) << " DEL: " << sqrt(log(1/delta)) << " EPS: " << epsilon << " POINTS: " << points.size() << std::endl;
            std::bernoulli_distribution bdist(0.5);
            std::exponential_distribution<double> edist(1 / (FULL_RANGE * sqrt(d) / (points.size() * epsilon)));
            std::exponential_distribution<double> g1_noise(1 / (5 / epsilon));
            std::uniform_real_distribution<double> total_random(0.0, 1.0);

            double size_noise = points.size() + g1_noise(gen) * (2 * bdist(gen) - 1) - (5 / epsilon) * log(2 / delta);
            // std::cout << points.size() << " " << size_noise << " " << (4 * FULL_RANGE / ((size_noise + 1) * epsilon)) * sqrt(2 * log(4 / delta)) << " " << gauss_variation<< std::endl;
            //double laplace_noise = points.size() + edist(gen) * bdist(gen);
            //double gauss_noise = points.size() + dist(gen) * gauss_variation;
            //double noise_ratio = (gauss_noise >= 0 ? gauss_noise : 0) / (laplace_noise >= 1 ? laplace_noise : 1);
            for (size_t i = 0; i < d; i++) {
                if (dist_type == 'G') {
                    if (size_noise <= 0 || isinf(gauss_variation)) {
                        //std::cout << "  -- NOISE SIZE " << points.size() << " " << size_noise << std::endl;
                        dest[i] = total_random(gen) * max_side;
                    } else {
                        //std::cout << " NOT NOISE " << points.size() << " " << gauss_variation << " " << (4 * FULL_RANGE / ((size_noise + 1) * epsilon)) * sqrt(2 * log(4 / delta)) << " " << size_noise << std::endl;
                        gauss_variation = (5 * FULL_RANGE / (4 * (size_noise) * epsilon)) * sqrt(2 * log(3.5 / delta));
                        dest[i] += dist(gen) * gauss_variation;//noise_ratio;//dist(gen) * gauss_variation;
                    }

                }
                else {
                    dest[i] += (bdist(gen) * 2 - 1) * edist(gen);
                }
//                double val = points.size() + edist(gen) * bdist(gen);
//                double denom = val <= 0 ? 1 : val;
//                double num = dest[i];
//                if (num > max_side) {
//                    num = max_side;
//                } else if (num < min_side) {
//                    num = min_side;
//                }
//                dest[i] = num / denom;
            }

        }


        /**
         * @brief Computers the average vector of the given list of vectors and inserts the new points into the given 
         * vector @param av_point.
         * 
         * @param points 
         * @param av_point 
         */
        void vaverage(const std::vector<std::vector<double>*>& points, std::vector<double>& av_point) {
            for (int j = 0; j < av_point.size(); j++) {
                double sum = 0;
                for (int i = 0; i < points.size(); i++) {
                    sum += points[i]->at(j);
                }
                av_point[j] = sum / (points.size() * 1.0);
            }
        }

        /**
         * @brief Calculates the l2 distance between two points.
         * 
         * @param v1 
         * @param v2 
         * @return double 
         */
        double l2_dist(const std::vector<double>& v1, const std::vector<double>& v2) {
            double sum = 0;
            for (int i = 0; i < v1.size(); i++) {
                sum += pow(v1[i] - v2[i], 2);
            }
            return sum;
        }

        /**
         * @brief Finds 
         * 
         * @param points 
         * @param chosen_assignments 
         * @param rev_assignments 
         * @return true 
         * @return false 
         */
        bool lloydassign(const std::vector<std::vector<double>*>& points, std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>& chosen_assignments, std::unordered_map<std::vector<double>*, std::vector<double>*>& rev_assignments) {
            bool change = false;
            for (int i = 0; i < points.size(); i++) {
                std::vector<double>* closest = nullptr;
                double sum = 0;
                for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = chosen_assignments.begin(); it != chosen_assignments.end(); it++) {
                    double temp_sum = sqrt(l2_dist(*it->first, *points[i]));
                    if (temp_sum < sum || closest == nullptr) {
                        sum = temp_sum;
                        closest = it->first;
                    }
                }
                chosen_assignments[closest]->push_back(points[i]);
                if (rev_assignments[points[i]] != closest) {
                    change = true;
                }
                rev_assignments[points[i]] = closest;
            }
            return change;
        }

        /**
         * @brief Lloyd's k-means clustering algorithm.
         * 
         * @param points 
         * @param k 
         * @param cycles 
         * @return std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>* 
         */
        std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>* lloydsalgo(const std::vector<std::vector<double>*>& points, int k, int cycles) {
            std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>& cluster_assignments = *(new std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>());
            std::vector<std::vector<double>*> ocenters;
            std::unordered_map<std::vector<double>*, std::vector<double>*> rev_assignments;

            if (points.size() < k) {
                k = points.size();
            }

            std::sample(points.begin(), points.end(), std::back_inserter(ocenters), k, gen);
            for (int i = 0; i < k; i++) {
                ocenters[i] = new std::vector<double>(*ocenters[i]);
                cluster_assignments[ocenters[i]] = new std::vector<std::vector<double>*>();
            }

            lloydassign(points, cluster_assignments, rev_assignments);

            bool change = true;
            int curr_cycle = 0;
            while (change) {
                std::cout << "ROUND " << curr_cycle << " OF LLOYDS" << std::endl;
                for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = cluster_assignments.begin(); it != cluster_assignments.end(); it++) {
                    if (it->second->size() == 0) {
                        continue;
                    }
                    vaverage(*it->second, *it->first);
                    it->second->clear();
                }

                change = lloydassign(points, cluster_assignments, rev_assignments);
                if (cycles > 0) {
                    if (curr_cycle >= cycles) {
                        break;
                    }
                }
                curr_cycle++;
            }
            return &cluster_assignments;
        }



        /**
         * @brief Lloyd's k-means clustering algorithm for weighted sets.
         * 
         * @param points 
         * @param k 
         * @param cycles 
         * @return std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>* 
         */
        std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>* wlloydsalgo(std::unordered_map<std::vector<double>*, int>& points, int k, int cycles) {
            std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>& cluster_assignments = *(new std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>());
            std::vector<std::vector<double>*> ocenters;
            std::unordered_map<std::vector<double>*, std::vector<double>*> rev_assignments;

            if (points.size() < k) {
                k = points.size();
            }
            std::vector<std::vector<double>*> keys;
            for (std::unordered_map<std::vector<double>*, int>::iterator it = points.begin(); it != points.end(); it++) {
                keys.push_back(it->first);
            }

            std::sample(keys.begin(), keys.end(), std::back_inserter(ocenters), k, gen);
            for (int i = 0; i < k; i++) {
                ocenters[i] = new std::vector<double>(*ocenters[i]);
                cluster_assignments[ocenters[i]] = new std::vector<std::vector<double>*>();
            }

            lloydassign(keys, cluster_assignments, rev_assignments);

            bool change = true;
            int curr_cycle = 0;
            while (change) {
                // std::cout << "NEW ROUND " << std::endl;
                for (std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>::iterator it = cluster_assignments.begin(); it != cluster_assignments.end(); it++) {
                    if (it->second->size() == 0) {
                        continue;
                    }
                    vaverage(*it->second, points, *it->first);
                    // std::cout << join(*it->first) << std::endl;
                    it->second->clear();
                }

                change = lloydassign(keys, cluster_assignments, rev_assignments);
                if (cycles > 0) {
                    if (curr_cycle >= cycles) {
                        break;
                    }
                }
            }
            std::cout << "WLLOYD FINISHED " << std::endl;
            return &cluster_assignments;
        }

        void vaverage(std::vector<std::vector<double>*>& points, std::unordered_map<std::vector<double>*, int>& weights, std::vector<double>& dest) {
            if (points.size() == 0) {
                return;
            }
            int d = points[0]->size();
            for (int i = 0; i < d; i++) {
                int total_size = 0;
                double sum = 0;
                for (int s = 0; s < points.size(); s++) {
                    sum += points[s]->at(i) * weights[points[s]];
                    total_size += weights[points[s]];
                }
                dest[i] = sum / total_size;
            }
        }

        /**
         * @brief Transforms the data by applying the JL transformation to embed all points
         * into the subpsace of @param new_dim dimensions and normalizes all the vectors.
         * 
         * @param data 
         * @param new_dim 
         * @return std::vector<std::vector<double>*>* 
         */
        std::vector<std::vector<double>*>* transform_data(const std::vector<std::vector<double>*>& data, int new_dim) {
            std::cout << "   TRANSFORMING DATA" << std::endl;
            std::vector<std::vector<double>*>* new_vec = new std::vector<std::vector<double>*>(data.size());
            int old_dim = data[0]->size();
            if (old_dim <= 5) {
                new_dim = old_dim;
            }
            double jl_matrix[new_dim][old_dim];
            std::bernoulli_distribution bunif(.5);
            for (int i = 0; i < new_dim; i++) {
                for (int j = 0; j < old_dim; j++) {
                    if (old_dim <= 5) {
                        if (i == j) {
                            jl_matrix[i][j] = 1;
                        } else {
                            jl_matrix[i][j] = 0;
                        }
                    } else {
                        jl_matrix[i][j] = (bunif(gen) * 2.0 - 1.0) / (sqrt(new_dim) * 3/2);
                    }
                }
            }
            std::cout << "   FINISHED JL MATRIX" << std::endl;
            double max = 0;
            for (int i = 0; i < new_vec->size(); i++) {
                std::vector<double>* npoint = new std::vector<double>(new_dim);
                std::vector<double>* opoint = data[i];
                double normalizer = 0;
                for (int j = 0; j < new_dim; j++) {
                    double sum = 0;
                    for (int k = 0; k < old_dim; k++) {
                        sum += opoint->at(k) * jl_matrix[j][k];
                    }
                    normalizer += pow(sum, 2);
                    npoint->operator[](j) = sum;
                }
                // std::cout << normalizer << std::endl;
                normalizer = sqrt(normalizer);
                if (normalizer > max) {
                    max = normalizer;
                }
                // for (int k = 0; k < new_dim; k++) {
                //     npoint->operator[](k) = npoint->operator[](k); // / normalizer;
                // }
                new_vec->operator[](i) = npoint;
            }
            for (int i = 0; i < new_vec->size(); i++) {
                for (int k = 0; k < new_dim; k++) {
                    new_vec->at(i)->operator[](k) = new_vec->at(i)->operator[](k) / max;
                }
                // std::cout << join(*new_vec->at(i)) << std::endl;
            }
            return new_vec;
        }




        /*
        NEW ITERATION USING LIST AND ITERATORS
        */


       /**
         * @brief Get the candidate centers object
         * 
         * @param m_data The data points
         * @param k The number of centers to choose.
         * @param d The dimension
         * @param epsilon The privacy epsilon parameter.
         * @param reservoir Where to store the chosen candidate centers.
         * @param chosen_assignments Map of candidate centers to the index of points assigned to them. 
         * @param r The radius.
         */
        void get_candidate_centers(List& m_data, int k, int d, double epsilon, std::vector<std::vector<double>*>& reservoir, std::unordered_map<std::vector<double>*, std::vector<std::vector<double>*>*>& chosen_assignments, double r, double rad) {
            if (m_data.size() == 0) {
                return;
            }
            std::vector<std::vector<int>*>* offset_mat = square_sum_generator(d, static_cast<int>(pow(rad + r * sqrt(d), 2)));
            
            std::cout << "    Finding Candidate Centers with radius, epsilon: " << r << ", " << epsilon << std::endl;

            std::unordered_map<std::string, int> tass;
            std::vector<int> acc(d);
            std::vector<std::vector<int>*> reprv;
            std::unordered_map<std::string, std::unordered_map<NodeList*, int>> assignments;
            std::unordered_map<std::vector<double>*, std::vector<std::string>> point_centers;
            std::unordered_map<std::string, int> indices_rev;
            NodeList* curr = m_data.begin;
            while (curr != nullptr) {
                for (int j = 0; j < offset_mat->size(); j++) {
                    capture_centers_it(curr, *offset_mat->at(j), tass, 0, acc, static_cast<int>(ceil(1 / r)), r, assignments, reprv, point_centers);
                }
                tass.clear();
                curr = curr->next;
            }
            // std::cout << "REPRV SIZE: " << reprv.size() << std::endl;
            // std::vector<double> unif_prob(reprv.size());
            // for (int i = 0; i < unif_prob.size(); i++) {
            //     indices_rev[join(*reprv[i])] = i;
            //     unif_prob[i] = 1;
            // }
            // int unif_total = reprv.size();
            int points_taken = 0;
            int ind;
            for (int round = 0; round < k; round++) {
                // std::cout << round << std::endl;
                if (m_data.size() == 0) {
                    for (int i = 0; i < offset_mat->size(); i++) {
                        delete offset_mat->at(i);
                    }
                    delete offset_mat;
                    for (int i = 0; i < reprv.size(); i++) {
                        delete reprv[i];
                    }
                    return;
                }


                int ind_max = 0;
                int max_assigns = 0;
                for (int i = 0; i < reprv.size(); i++) {
                    std::string vcs = join(*reprv[i]);
                    if (assignments[vcs].size() > max_assigns) {
                        max_assigns = assignments[vcs].size();
                        ind_max = i;
                    }
                }
                double total_size = 0;
                std::vector<double> probs(reprv.size());

                double tval;
                int subtractions = 0;
                for (int i = 0; i < reprv.size(); i++) {
                    std::string vcs = join(*reprv[i]);
                    if (assignments[vcs].size() == 0) {
                        probs[i] = 0;
                        subtractions += 1;
                        continue;
                    }
                    tval = exp(epsilon * (static_cast<int>(assignments[vcs].size()) - max_assigns) / 2.0);
                    probs[i] = tval - 1 / exp(epsilon * max_assigns / 2);
                    probs[i] = probs[i] < 0 ? 0 : probs[i];
                    if (probs[i] == 0) {
                        subtractions += 1;
                    }
                    total_size += probs[i];
                }
//                int i = 0;
//                for (std::unordered_map<std::string, std::unordered_map<NodeList*, int>>::iterator it = assignments.begin(); it != assignments.end(); it++) {
//                    double adder = 0;
//                    if (it->second.size() != 0) {
//                        adder = exp(epsilon * (static_cast<int>(it->second.size()) - max_assigns) / 2.0);;
//                    }
//                    probs
//                    total_size += adder;
//                    i++;
//                }

                std::bernoulli_distribution bdist(1 - pow(1/r, d) / (total_size * pow(1/r, d) - (static_cast<int>(reprv.size()) - subtractions)));
                // std::cout << 1 - pow(1/r, d) / (total_size + pow(1/r, d) - (static_cast<int>(reprv.size()) - subtractions)) << " " << total_size << std::endl;
                if (bdist(gen)) {
                    // std::cout << max_assigns << std::endl;
                    ind = sample(probs, total_size);
                    // std::cout << " WOULD HAVE CHOSEN " << ind << " PROBABILITY CHOSEN: " << std::setprecision(std::numeric_limits<double>::digits10 + 2) << probs[ind] << std::endl;
                } else {
                    //std::cout << "      Uniform Attempt" << std::endl;
                    std::uniform_int_distribution unif_dist(-static_cast<int>(1 / r), static_cast<int>(1 / r));
                    std::vector<int> complement(d);
                    for (int tolerance = 0; tolerance < 100; tolerance++) {
                        for (int dim = 0; dim < d; dim++) {
                            complement[dim] = unif_dist(gen);
                        }

                        std::string complement_string = join(complement);
                        if (assignments.find(complement_string) == assignments.end()) {
                            std::vector<double>* true_vec = new std::vector<double>();
                            for (int dim = 0; dim < d; dim++) {
                                true_vec->push_back(complement[dim] * r);
                            }
                            reservoir.push_back(true_vec);
                            chosen_assignments[true_vec] = new std::vector<std::vector<double>*>();
                            break;
                        }
                        if (tolerance == 99) {
                            std::cout << "Could not find complement to add" << std::endl;
                        }
                    }
                    continue;
                    //ind = sample(unif_prob, unif_total);
                }
                //ind = ind_max;
                //std::cout << "     INDEX CHOSEN " << ind << std::endl;

                // unif_prob[ind] = 0;
                // unif_total--;
                std::vector<int>* cand_center_to_add = reprv[ind];
                std::vector<double>* n_center = new std::vector<double>();
                for (int i = 0; i < cand_center_to_add->size(); i++) {
                    n_center->push_back(cand_center_to_add->at(i) * r);
                }
                reservoir.push_back(n_center);
                std::string cand_center = join(*cand_center_to_add);
                std::unordered_map<NodeList*, int>& ccenter = assignments[cand_center];
                int curix = 0;
                std::vector<std::vector<double>*>* nadd = new std::vector<std::vector<double>*>(ccenter.size());
                for (std::unordered_map<NodeList*, int>::iterator it = ccenter.begin(); it != ccenter.end(); it++) {
                    points_taken++;
                    m_data.remove(it->first);
                    nadd->operator[](curix) = it->first->point;
                    std::vector<std::string>& vals = point_centers[it->first->point];
                    for (int s = 0; s < vals.size(); s++) {
                        std::unordered_map<NodeList*, int>& tempnl = assignments[vals[s]];
                        if (&tempnl == &ccenter) {
                            continue;
                        }
                        tempnl.erase(tempnl.find(it->first));
                        // if (tempnl.size() == 0) {
                        //     unif_prob[indices_rev[vals[s]]] -= 1;
                        //     unif_total--;
                        // }
                    }
                    curix++;
                }
                
                ccenter.clear();
                chosen_assignments[n_center] = nadd;
            }
            // std::cout << "      POINTS REMOVED: " << points_taken << std::endl;
            for (int i = 0; i < reprv.size(); i++) {
                delete reprv[i];
            }
            for (int i = 0; i < offset_mat->size(); i++) {
                delete offset_mat->at(i);
            }
            delete offset_mat;
        }





         /**
         * @brief Given a point, finds all grid points that are only an offset away from the given point.
         * 
         * @param point The point.
         * @param offsets The offset to consider.
         * @param assignments Map to check if we have already encountered a point.
         * @param index The current index we are considering.
         * @param reprv The vector holding all new grid points.
         * @param acc The grid point we are currently building.
         * @param limit The maximium grid point a point may take.
         * @param r The current side length of the net.
         * @param tassignments Global assignments.
         */
        void capture_centers_it(NodeList* point, std::vector<int>& offsets, std::unordered_map<std::string, int>& assignments, int index, std::vector<int>& acc, int limit, double r, std::unordered_map<std::string, std::unordered_map<NodeList*, int>>& tassignments, std::vector<std::vector<int>*>& treprv, std::unordered_map<std::vector<double>*, std::vector<std::string>>& point_centers) {
            if (index == point->point->size() - 1) {
                std::vector<int>* temp;
                for (int j = 0; j < 2; j++) {
                    offsets[index] *= -1;
                    if (offsets[index] >= 0) {
                        if (offsets[index] > 0 || j == 0) {
                            acc[index] = static_cast<int>(ceil(point->point->operator[](index) / r)) + offsets[index];
                            if (acc[index] <= -limit) {
                                acc[index] = -limit;
                            } else if (acc[index] >= limit) {
                                acc[index] = limit;
                            }
                        }
                    }
                    if (offsets[index] <= 0) {
                        if (offsets[index] < 0 || j == 1) {
                            acc[index] = static_cast<int>(floor(point->point->operator[](index) / r)) + offsets[index];
                            if (acc[index] <= -limit) {
                                acc[index] = -limit;
                            } else if (acc[index] >= limit) {
                                acc[index] = limit;
                            }
                        }
                    }
                    std::string vc = join(acc);
                    if (assignments.find(vc) == assignments.end()) {
                        assignments[vc] = -1;
                        if (tassignments.find(vc) == tassignments.end()) {
                            temp = new std::vector<int>(acc);
                            treprv.push_back(temp);
                            tassignments[vc] = std::unordered_map<NodeList*, int>();
                        }
                        tassignments[vc][point] = -1;
                        point_centers[point->point].push_back(vc);
                    }
                }
                return;
            }
            
            
            offsets[index] *= -1;
            if (offsets[index] <= 0) {
                acc[index] = static_cast<int>(floor(point->point->operator[](index) / r) + offsets[index]);
                if (acc[index] <= -limit) {
                    acc[index] = -limit;
                } else if (acc[index] >= limit) {
                    acc[index] = limit;
                }
                capture_centers_it(point, offsets, assignments, index + 1, acc, limit, r, tassignments, treprv, point_centers);
            }
            offsets[index] *= -1;
            if (offsets[index] >= 0) {
                acc[index] = static_cast<int>(ceil(point->point->operator[](index) / r)) + offsets[index];
                if (acc[index] <= -limit) {
                    acc[index] = -limit;
                } else if (acc[index] >= limit) {
                    acc[index] = limit;
                }
                capture_centers_it(point, offsets, assignments, index + 1, acc, limit, r, tassignments, treprv, point_centers);
            }
        }

};
