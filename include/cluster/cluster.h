#ifndef CLUSTER_H
#define CLUSTER_H

#include <algorithm> 
#include <cassert>
#include <chrono>
#include <cmath> 
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <random> 
#include <string>
#include <vector>

#include "../cluster/cluster_utils.h"
#include "../modules/exact_nn/exact_nn.h"
#include "../modules/hypercube/hypercube.h"
#include "../modules/lsh/lsh.h"

#define EPSILON 500000
#define CLSH 1.8

template <typename T> class Cluster
{

  private:
    
    const size_t ClustersNUM;

    

    std::vector<std::vector<size_t>> clusters;

    std::vector<std::vector<T>> centroids;

    std::vector<double> avg_sk;

    double Stotal = 0.0;
    LSH<T> *LSH_ptr;
    Hypercube<T> *CUBE_ptr;

  public:
    // Lloyds Assignment 
    Cluster(size_t nclusters) : ClustersNUM(nclusters), LSH_ptr(nullptr), CUBE_ptr(nullptr)
    {
        clusters.resize(ClustersNUM);
        avg_sk.resize(ClustersNUM, 0.0);
    }

    Cluster(size_t nclusters, uint16_t L, uint16_t N, uint32_t K, double mean_dist,
            const std::vector<std::vector<T>> &trainingset)
        : ClustersNUM(nclusters), CUBE_ptr(nullptr)
    {
        clusters.resize(ClustersNUM);
        avg_sk.resize(ClustersNUM, 0.0);
        LSH_ptr = new LSH<T>(L, N, K, mean_dist, trainingset);
    }

    Cluster(size_t nclusters, uint32_t cube_dims, uint16_t M, uint16_t probes, uint16_t N, float R, size_t trainingSIZE,
            uint32_t DATAdims, double meanNN_dist, const std::vector<std::vector<T>> &trainingset)
        : ClustersNUM(nclusters), LSH_ptr(nullptr)
    {
        clusters.resize(ClustersNUM);
        avg_sk.resize(ClustersNUM, 0.0);
        CUBE_ptr = new Hypercube<T>(cube_dims, M, probes, N, R, trainingSIZE, DATAdims, meanNN_dist, trainingset);
    }

    ~Cluster()
    {
        if (LSH_ptr != nullptr)
            delete LSH_ptr;
        if (CUBE_ptr != nullptr)
            delete CUBE_ptr;
    }

    void INITplusplus(const std::vector<std::vector<T>> &trainingSET, std::vector<size_t> &centroid_indexes)
    {
        std::vector<std::pair<float, size_t>> partialSUMS;
        std::vector<float> MINdistance(trainingSET.size());

  
        std::default_random_engine generator;
        srand((unsigned)time(NULL));
        size_t size = trainingSET.size();
        size_t index = rand() % size;


        centroids.emplace_back(trainingSET[index]);

        centroid_indexes[0] = index;

        for (size_t t = 1; t != ClustersNUM; ++t)
        {

            for (size_t i = 0; i != size; ++i)
            {


                if (in(centroid_indexes, i))
                    continue;

                MINdistance[i] = exact_nn<T>(centroids, trainingSET[i]);
            }
             NormalizeDist(MINdistance);

 
            float prev_partial_sum = 0.0;
            float new_partial_sum = 0.0;
            partialSUMS.emplace_back(0.0, 0); 
            for (size_t j = 0; j != size; ++j)
            {

                if (in(centroid_indexes, j))
                    continue;

                new_partial_sum =
                    prev_partial_sum + (MINdistance[j] * MINdistance[j]);
                partialSUMS.emplace_back(new_partial_sum, j);
                prev_partial_sum = new_partial_sum;
            }

            std::uniform_real_distribution<float> distribution(0.0, new_partial_sum);
            float x = distribution(generator);
            std::sort(partialSUMS.begin(), partialSUMS.end(), CompareF);
            size_t r = BinarySearch(partialSUMS, x);

            centroids.emplace_back(trainingSET[r]);

            centroid_indexes[t] = r;

            partialSUMS.clear();
        }
    }

    /* lloyd's assignment */
    void LLOYS_assign(const std::vector<std::vector<T>> &train_set, std::vector<size_t> &centroid_indexes)
    {
        uint32_t min_dist{};
        uint32_t dist{};

    
        for (size_t i = 0; i != train_set.size(); ++i)
        {
            
            if (in(centroid_indexes, i))
                continue;
            min_dist = std::numeric_limits<uint32_t>::max();
            size_t best_centroid{};
            for (size_t j = 0; j != centroids.size(); ++j)
            {
                dist = euclidean_distance_rd<T>(train_set[i], centroids[j]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_centroid = j;
                }
            }
            clusters[best_centroid].emplace_back(i);
        }
    }

    void LLOYS_assign(const std::vector<std::vector<T>> &train_set)
    {
        uint32_t min_dist{};
        uint32_t dist{};




        for (size_t i = 0; i != train_set.size(); ++i)
        {
            min_dist = std::numeric_limits<uint32_t>::max();
            size_t best_centroid{};
            for (size_t j = 0; j != centroids.size(); ++j)
            {
                dist = euclidean_distance_rd<T>(train_set[i], centroids[j]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_centroid = j;
                }
            }
    


            clusters[best_centroid].emplace_back(i);
        }
    }

    void REVERSE_assign(const std::vector<std::vector<T>> &train_set, std::vector<size_t> &centroid_indexes)
    {
        assert(centroids.size() == ClustersNUM);

        std::map<int, int> assigned_vectors;
        ssize_t n_vectors = train_set.size();
        ssize_t n_centroids = centroids.size();

        for (ssize_t i = 0; i != n_vectors; ++i)
        {
            assigned_vectors[i] = -1;
        }

        uint32_t dist{};
        uint32_t min_dist = std::numeric_limits<uint32_t>::max();
        for (ssize_t i = 0; i != n_centroids; ++i)
        {
            for (ssize_t j = i + 1; j != n_centroids; ++j)
            {
                dist = euclidean_distance_rd<T>(centroids[i], centroids[j]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                }
            }
        }

        double RADIUS = (double)(min_dist / 2);

        size_t new_assigned = 0;
        std::vector<size_t> range_search_nns;

        while (1)
        {

            for (ssize_t i = 0; i != n_centroids; ++i)
            {

                if (CUBE_ptr == nullptr)
                {
                    range_search_nns = LSH_ptr->approximate_range_search(CLSH, RADIUS, centroids[i]);
                }
                else
                {
                    range_search_nns = CUBE_ptr->range_search(centroids[i], train_set, RADIUS);
                }
                for (const auto &vector_index : range_search_nns)
                {

                    if (in(centroid_indexes, vector_index))
                        continue;


                    if (assigned_vectors[vector_index] == -1)
                    {
                        clusters[i].emplace_back(vector_index);
                      
                        assigned_vectors[vector_index] = i;
                        ++new_assigned;
                    }

                    else if (assigned_vectors[vector_index] != i)
                    {
                        int assigned_centroid = assigned_vectors[vector_index];
                        uint32_t prev_centroid_dist =
                            euclidean_distance_rd<T>(train_set[vector_index], centroids[assigned_centroid]);
                        uint32_t new_centroid_dist = euclidean_distance_rd<T>(train_set[vector_index], centroids[i]);


                        if (new_centroid_dist < prev_centroid_dist)
                        {
                            ++new_assigned;

                            for (auto iter = clusters[assigned_centroid].begin();
                                 iter != clusters[assigned_centroid].end(); ++iter)
                            {
                                if (*iter == vector_index)
                                {
                                    clusters[assigned_centroid].erase(iter);
                                    break;
                                }
                            }

                            clusters[i].emplace_back(vector_index);


                            assigned_vectors[vector_index] = i;
                        }
                    }
                }
            }

            if (RADIUS > 20000.0 && new_assigned == 0)
                break;

            new_assigned = 0;

            RADIUS *= 2;
        }

        for (ssize_t i = 0; i != n_vectors; ++i)
        {
            if (assigned_vectors[i] == -1)
            {
                min_dist = std::numeric_limits<uint32_t>::max();
                int best_centroid{};
                for (ssize_t j = 0; j != n_centroids; ++j)
                {
                    uint32_t dist = euclidean_distance_rd(train_set[i], centroids[j]);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        best_centroid = j;
                    }
                }
                clusters[best_centroid].emplace_back(i);
            }
        }
    }

    void REVERSE_assign(const std::vector<std::vector<T>> &train_set)
    {
        assert(centroids.size() == ClustersNUM);

        std::map<int, int> assigned_vectors;
        ssize_t n_vectors = train_set.size();
        ssize_t n_centroids = centroids.size();


        for (ssize_t i = 0; i != n_vectors; ++i)
        {
            assigned_vectors[i] = -1;
        }


        uint32_t dist{};
        uint32_t min_dist = std::numeric_limits<uint32_t>::max();
        for (ssize_t i = 0; i != n_centroids; ++i)
        {
            for (ssize_t j = i + 1; j != n_centroids; ++j)
            {
                dist = euclidean_distance_rd<T>(centroids[i], centroids[j]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                }
            }
        }


        double RADIUS = (double)(min_dist / 2);
        size_t new_assigned = 0;
        std::vector<size_t> range_search_nns;

        while (1)
        {

            for (ssize_t i = 0; i != n_centroids; ++i)
            {

                if (CUBE_ptr == nullptr)
                {
                    range_search_nns = LSH_ptr->approximate_range_search(CLSH, RADIUS, centroids[i]);
                }
                else
                {
                    range_search_nns = CUBE_ptr->range_search(centroids[i], train_set, RADIUS);
                }
                for (const auto &vector_index : range_search_nns)
                {

            
                    if (assigned_vectors[vector_index] == -1)
                    {
                        clusters[i].emplace_back(vector_index);
                        
                        assigned_vectors[vector_index] = i;
                        ++new_assigned;
                    }

                    
                    else if (assigned_vectors[vector_index] != i)
                    { 
                        int assigned_centroid = assigned_vectors[vector_index];
                        uint32_t prev_centroid_dist =
                            euclidean_distance_rd<T>(train_set[vector_index], centroids[assigned_centroid]);
                        uint32_t new_centroid_dist = euclidean_distance_rd<T>(train_set[vector_index], centroids[i]);

                        if (new_centroid_dist < prev_centroid_dist)
                        {
                            ++new_assigned;


                            for (auto iter = clusters[assigned_centroid].begin();
                                 iter != clusters[assigned_centroid].end(); ++iter)
                            {
                                if (*iter == vector_index)
                                {
                                    clusters[assigned_centroid].erase(iter);
                                    break;
                                }
                            }

                            clusters[i].emplace_back(vector_index);

                            assigned_vectors[vector_index] = i;
                        }
                    }
                }
            }

            if (RADIUS > 20000.0 && new_assigned == 0)
                break;

            RADIUS *= 2;
            new_assigned = 0;
        }


        for (ssize_t i = 0; i != n_vectors; ++i)
        {
            if (assigned_vectors[i] == -1)
            {
                min_dist = std::numeric_limits<uint32_t>::max();
                int best_centroid{};
                for (ssize_t j = 0; j != n_centroids; ++j)
                {
                    uint32_t dist = euclidean_distance_rd(train_set[i], centroids[j]);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        best_centroid = j;
                    }
                }
                clusters[best_centroid].emplace_back(i);
            }
        }
    }

    void medianUPDATE(const std::vector<std::vector<T>> &train_set)
    {
        assert(centroids.size() == clusters.size());

        const size_t dim = centroids[0].size();
        std::vector<T> components;

        for (size_t k = 0; k != ClustersNUM; ++k)
        {

            std::vector<T> &k_centroid = centroids[k];
            size_t cluster_size = clusters[k].size();
            components.resize(cluster_size);
            const std::vector<size_t> &cluster_indexes = clusters[k];

            for (size_t d = 0; d != dim; ++d)
            {

                for (size_t t = 0; t != cluster_size; ++t)
                {

                    const std::vector<T> &t_vector = train_set[cluster_indexes[t]];
                    components[t] = t_vector[d];
                }
                std::sort(components.begin(), components.end());
                size_t median_index = std::ceil(cluster_size / 2);
                k_centroid[d] = components[median_index];
            }
        }
    }

    uint64_t objective_function(const std::vector<std::vector<T>> &train_set)
    {
        size_t size = train_set.size();
        uint32_t min_dist = 0;
        uint64_t l1_norm = 0;

        for (size_t i = 0; i != size; ++i)
        {
            min_dist = exact_nn<T>(centroids, train_set[i]);
            l1_norm += min_dist;
        }

        return l1_norm;
    }

    void k_medians_plus_plus(const std::vector<std::vector<T>> &train_set, const std::string &METHOD)
    {
        long prev_objective = 0;
        long new_objective = 0;
        std::vector<size_t> centroid_indexes(ClustersNUM);

        INITplusplus(train_set, centroid_indexes);

        
        if (METHOD == "Classic")
            LLOYS_assign(train_set, centroid_indexes);
        else
            REVERSE_assign(train_set, centroid_indexes);

        // step 2: median update
        medianUPDATE(train_set);

        for (auto &cluster : clusters)
        {
            cluster.clear();
        }


        while (1)
        {


            if (METHOD == "Classic")
                LLOYS_assign(train_set);
            else
                REVERSE_assign(train_set);


            medianUPDATE(train_set);


            new_objective = objective_function(train_set);

            std::cout << "\nObjective of n-1 is " << prev_objective << std::endl;
            std::cout << "Objective of n   is " << new_objective << std::endl;


            if (std::abs(prev_objective - new_objective) < EPSILON)
                break;


            for (auto &cluster : clusters)
            {
                cluster.clear();
            }

            prev_objective = new_objective;
        }
    }

    void silhouette(const std::vector<std::vector<T>> &dataset)
    {
        const size_t n_vectors = dataset.size();

        std::vector<double> s(n_vectors);
        std::vector<double> a(n_vectors);
        std::vector<double> b(n_vectors);

        /* compute a[i] values */
        for (auto it = clusters.cbegin(); it != clusters.cend(); ++it)
        {
            const std::vector<size_t> &each_cluster_vector_indexes = *it; // reference instead of copying it
            for (size_t i = 0; i != each_cluster_vector_indexes.size(); ++i)
            {
                size_t total_a_dist{};
                for (size_t j = 0; j != each_cluster_vector_indexes.size(); ++j)
                {
                    if (i == j)
                        continue;
                    total_a_dist += euclidean_distance_rd<T>(dataset[each_cluster_vector_indexes[i]],
                                                             dataset[each_cluster_vector_indexes[j]]);
                }
                if (each_cluster_vector_indexes.size() > 1)
                {
                    a[each_cluster_vector_indexes[i]] = (double)total_a_dist / each_cluster_vector_indexes.size();
                }
                else
                {
                    a[each_cluster_vector_indexes[i]] = (double)total_a_dist; // in this case a[i] = 0
                }
            }
        }

        /* compute closest centroid to each centroid */
        std::vector<size_t> closest_centroids(centroids.size());
        for (size_t i = 0; i != centroids.size(); ++i)
        {
            uint32_t min_dist = std::numeric_limits<uint32_t>::max();
            size_t closest = 0;
            for (size_t j = 0; j != centroids.size(); ++j)
            {
                if (i == j)
                    continue;
                uint32_t dist = euclidean_distance_rd<T>(centroids[i], centroids[j]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    closest = j;
                }
            }
            closest_centroids[i] = closest; 
        }

        /* compute b[i] values */
        for (size_t k = 0; k != clusters.size(); ++k)
        {
            const std::vector<size_t> &each_cluster_vector_indexes = clusters[k];
            const std::vector<size_t> &closest_cluster_vector_indexes = clusters[closest_centroids[k]];
            for (size_t i = 0; i != each_cluster_vector_indexes.size(); ++i)
            {
                size_t total_b_dist{};
                for (size_t j = 0; j != closest_cluster_vector_indexes.size(); ++j)
                {
                    total_b_dist += euclidean_distance_rd<T>(dataset[each_cluster_vector_indexes[i]],
                                                             dataset[closest_cluster_vector_indexes[j]]);
                }
                if (closest_cluster_vector_indexes.size() > 0)
                {
                    b[each_cluster_vector_indexes[i]] = (double)total_b_dist / closest_cluster_vector_indexes.size();
                }
                else
                {
                    b[each_cluster_vector_indexes[i]] = (double)total_b_dist;
                }
            }
        }


        for (size_t i = 0; i != n_vectors; ++i)
        {
            s[i] = (b[i] - a[i]) / std::max(a[i], b[i]);
        }

        for (size_t i = 0; i != centroids.size(); ++i)
        {
            const std::vector<size_t> &each_cluster_vector_index = clusters[i];
            size_t n_vectors = each_cluster_vector_index.size();
            for (size_t j = 0; j != n_vectors; ++j)
            {
                avg_sk[i] += s[each_cluster_vector_index[j]];
            }
            if (n_vectors != 0)
            {
                avg_sk[i] /= n_vectors;
            }
        }

        uint32_t n_centroids = centroids.size();

        for (size_t i = 0; i != n_centroids; ++i)
        {
            Stotal += avg_sk[i];
        }
        Stotal /= n_centroids;
    }

    void ClusterOUTput(const std::string &out, const std::string &METHOD, bool complete,
                              std::chrono::seconds cluster_time)
    {
        std::ofstream ofile;
        ofile.open(out, std::ios::out | std::ios::trunc);

        if (ofile)
        {
            ofile << "Algorithm: ";
            if (METHOD == "Classic")
            {
                ofile << "Lloyds" << std::endl;
            }
            else if (METHOD == "LSH")
            {
                ofile << "Range Search LSH" << std::endl;
            }
            else
            {
                ofile << "Range Search Hypercube" << std::endl;
            }

            for (size_t i = 0; i != clusters.size(); ++i)
            {
                ofile << "CLUSTER-" << i + 1 << " {size: " << clusters[i].size() << ", centroid: [";
                for (auto &c : centroids[i])
                {
                    ofile << +c << " ";
                }
                ofile << "]}" << std::endl;
            }
            ofile << "clustering_time: " << std::chrono::duration<double>(cluster_time).count() << " seconds"
                  << std::endl;
            ofile << "Silhouette: [";
            for (auto &s : avg_sk)
            {
                ofile << s << ", ";
            }
            ofile << Stotal << "]\n\n" << std::endl;

            if (complete)
            {
                for (size_t i = 0; i != clusters.size(); ++i)
                {
                    ofile << "CLUSTER-" << i + 1 << " {[";
                    for (auto &c : centroids[i])
                    {
                        ofile << +c << " ";
                    }
                    ofile << "],";
                    for (auto &i : clusters[i])
                    {
                        ofile << " " << i;
                    }
                    ofile << "}" << std::endl;
                }
            }
        }
        else
        {
            std::cerr << "\nCould not open output file!\n" << std::endl;
        }
    }
};

#endif
