#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include <unordered_map>
#include <vector>
#include <utility>   
#include <limits>    
#include <algorithm> 
#include <cassert>
#include <cstdlib>
#include <cmath>

#include "../../hash_function/hash_function.h"
#include "../../metric/metric.h"

#define MULTIPLE1 4
#define MULTIPLE2 10
#define C      1.1


template <typename T>
class Hypercube {

    private:

        uint32_t m;
        uint32_t M;


        const uint32_t projection_dimension;
        const uint16_t max_candidates;
        const uint16_t max_probes;

        const uint16_t N;
        const float    R;

        const size_t   train_samples;
        const uint32_t D;
        const double   win;

        std::vector<std::unordered_multimap<uint32_t, bool>> uniform_binary_mapppings;

  
        std::unordered_map<std::string, std::vector<size_t>> hash_table;


        std::vector<HashFunction<T>> hash_functions;


        void init_k_nearest_neighbors(std::vector<std::pair<uint32_t, size_t>> &k_nearest)
        {
            uint32_t best_dist = std::numeric_limits<uint32_t>::max();

            for (size_t i = 0; i != max_candidates; ++i) {
                k_nearest.emplace_back(best_dist, 0);
            }
        }


    public:

        Hypercube (uint32_t projdim, uint16_t cands, uint16_t probes, uint16_t nns, float r, \
                    size_t trn, uint32_t d, double meandist, const std::vector<std::vector<T>> &samples) \
                    : projection_dimension(projdim), max_candidates(cands), max_probes(probes), \
                      N(nns), R(r), train_samples(trn), D(d), win(MULTIPLE2 * meandist)
        {
            std::cout << "Window is: ";
            std::cout << win << std::endl;

            M = 1ULL << (32 / projection_dimension);
            m = (1ULL << 32) - (5);

            std::cout << "M is " << M << std::endl;
            std::cout << "m is " << m << std::endl;


            for (size_t i = 0; i != projection_dimension; ++i) {
                hash_functions.emplace_back( HashFunction<T> (0, D, m, M, win) );
                uniform_binary_mapppings.emplace_back(std::unordered_multimap<uint32_t, bool>());
            }

            hash_table.rehash(trn); 
            for (size_t i = 0; i != train_samples; ++i) {
                cube_projection_train(samples[i], i);
            }
        }



        void cube_projection_train(const std::vector<T>& point, size_t index) {
            std::string bitstring;

            for (size_t j = 0; j < projection_dimension; ++j) {
                uint32_t hval = hash_functions[j].HashFunctionConstructor(point);

                // Check if f_j(h_j(p)) has already been computed
                short bit = retrieve_val(uniform_binary_mapppings[j], hval);

          
                if (bit == -1) {
                    bit = uniform_binmap(uniform_binary_mapppings[j], hval);
                }

                bitstring += std::to_string(bit);
            }

            assert(bitstring.size() == projection_dimension);


            hash_table[bitstring].push_back(index);
        }

  
        std::string cube_projection_test(const std::vector<T>& query) {
            std::string bitstring;

            for (size_t j = 0; j < projection_dimension; ++j) {
                uint32_t hval = hash_functions[j].HashFunctionConstructor(query);

        
                short bit = retrieve_val(uniform_binary_mapppings[j], hval);


                if (bit == -1) {
                    bit = uniform_binmap(uniform_binary_mapppings[j], hval);
                }

                bitstring += std::to_string(bit);
            }

            assert(bitstring.size() == projection_dimension);

            return bitstring;
        }

        short retrieve_val(const std::unordered_multimap<uint32_t, bool> &map, uint32_t hval)
        {
            auto it = map.find(hval);
            if (it != map.end()) {
                return it->second;
            } else {
                return -1;
            }
        }


        short uniform_binmap(std::unordered_multimap<uint32_t, bool> &map, uint32_t hval)
        {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist(0,1); 
            bool bit = dist(rng);  
            map.emplace(hval, bit); 

            return bit;
        }


        std::vector<std::pair<uint32_t, size_t>> approximateNN(const std::vector<T> &query, const std::vector<std::vector<T>> &train_samples)
        {
            std::vector<std::pair<uint32_t, size_t>> candidates(max_candidates);
            uint16_t probes = max_probes;
            std::string key = cube_projection_test(query);
            size_t bucket_count = 0;
            uint8_t bits = 1;
            uint32_t cnt = projection_dimension;

            init_k_nearest_neighbors(candidates);

            while (probes > 0 && M > 0) {
                bucket_count = hash_table[key].size();
                const std::vector<size_t> &indexes = hash_table[key]; 
                for (size_t i = 0; i < bucket_count && M > 0; ++i, --M) {
                    uint32_t dist = euclidean_distance_rd<T>(query, train_samples[indexes[i]]);
                    if (dist < candidates[0].first) {
                        candidates[0] = std::make_pair(dist, indexes[i]);
                        std::sort(candidates.begin(), candidates.end(), [](const std::pair<uint32_t, size_t> &left, \
                                                                            const std::pair<uint32_t, size_t> &right) \
                                                                            { return (left.first > right.first); } );
                    }
                }

                key = gen_nearby_vertex(key, cnt, bits);
                if (bits > projection_dimension) break;
                --probes;
            }

            std::sort(candidates.begin(), candidates.end(), [](const std::pair<uint32_t, size_t> &left, \
                                                                    const std::pair<uint32_t, size_t> &right) \
                                                                    { return (left.first < right.first); } );

            return candidates;
        }

        std::vector<size_t> range_search(const std::vector<T> &query, const std::vector<std::vector<T>> &train_samples, float r = 0.0)
                {
                    std::vector<size_t> candidates;
                    uint32_t dist = 0;
                    uint32_t cnt = projection_dimension;
                    uint16_t M = max_candidates;
                    uint16_t probes = max_probes;
                    uint8_t  bits = 1;
                    size_t   bucket_count = 0;

                    if (r == 0.0) 
                      r = R;

                    const std::string key = cube_projection_test(query);
                    std::string key1 = key;

                    while (M > 0 && probes > 0 && bits <= projection_dimension) {
                        bucket_count = hash_table[key1].size();
                        const std::vector<size_t> &indexes = hash_table[key1];
                        for (size_t i = 0; i < bucket_count && M > 0; ++i, --M) {
                            dist = euclidean_distance_rd<T>(query, train_samples[indexes[i]]);
                            if (dist < C * r) {
                                candidates.push_back(indexes[i]);
                            }
                        }

                        key1 = gen_nearby_vertex(key, cnt, bits);
                        ++bits;
                        --probes;
                    }

                    return candidates;
                }
        std::string gen_nearby_vertex(const std::string &key, uint32_t &counter, uint8_t &bits) {
            std::string bitstring = key;
            for (size_t i = 0; i < bits && counter != 0; ++i, --counter) {
                if (bitstring[counter - 1] == '0') {
                    bitstring[counter - 1] = '1';
                } else {
                    bitstring[counter - 1] = '0';
                }
                if (counter == 1 && i < bits) {
                    counter = projection_dimension;
                }
            }

            if (counter == 0) {
                counter = projection_dimension; 
                ++bits;
            }

            return bitstring;
        }
};

#endif
