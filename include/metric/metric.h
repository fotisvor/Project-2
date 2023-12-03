#ifndef METRIC_H
#define METRIC_H

#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <vector>

template <typename T>
inline double euclidean_distance_rd(const std::vector<T> &DATAset_vector, const std::vector<T> &QueryVector)
{
    assert(DATAset_vector.size() == QueryVector.size());

    double Distance{};
    auto d_it = DATAset_vector.cbegin();
    auto q_it = QueryVector.cbegin();

    for (; d_it != DATAset_vector.cend(); ++d_it, ++q_it)
    {
        Distance += std::pow((*d_it) - (*q_it), 2);
    }

    return std::sqrt(Distance);
}

template <typename T> inline uint32_t NN_distance(const std::vector<std::vector<T>> &dataset)
{
    uint32_t Distance{};
    uint32_t sum{};
    uint32_t bestDistance = std::numeric_limits<uint32_t>::max();

    for (size_t i = 0; i != dataset.size() / 600; ++i)
    {

        for (size_t j = 0; j != dataset.size() / 600; ++j)
        {

            if (i == j)
                continue;

            Distance = euclidean_distance_rd<T>(dataset[i], dataset[j]);
            if (Distance < bestDistance)
                bestDistance = Distance;
        }
        sum = bestDistance + sum;
        bestDistance = std::numeric_limits<uint32_t>::max();
    }
    return (sum / (dataset.size() / 60));
}

#endif //