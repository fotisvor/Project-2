#ifndef EXACT_NN
#define EXACT_NN

#include <vector>
#include <limits> 
#include <algorithm>

#include "../../../include/metric/metric.h"


template<typename T>
std::vector<uint32_t> exact_nn(const std::vector<std::vector<T>>& dataset, const std::vector<T>& query, uint16_t nns) {
    std::vector<uint32_t> closest_distances(nns, std::numeric_limits<uint32_t>::max());
    for (const auto& point : dataset) {
        const uint32_t dist = euclidean_distance_rd<T>(point, query);
        if (dist < closest_distances.back()) {
            closest_distances.back() = dist;
            std::sort(closest_distances.begin(), closest_distances.end());
        }
    }
    return closest_distances;
}


template<typename T>
uint32_t exact_nn(const std::vector<std::vector<T>>& centroids, const std::vector<T>& point) {
    uint32_t min_dist = std::numeric_limits<uint32_t>::max();
    for (const auto& centroid : centroids) {
        const uint32_t dist = euclidean_distance_rd<T>(centroid, point);
        min_dist = std::min(min_dist, dist);
    }
    return min_dist;
}

#endif // EXACT_NN
