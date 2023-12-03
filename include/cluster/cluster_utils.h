#ifndef CLUSTER_UTILS_H
#define CLUSTER_UTILS_H

#include <string>
#include <vector>
#include <utility>


typedef struct ClusterArgs {
    std::string InputFile;
    std::string config_file;
    std::string OutputFile;
    std::string METHOD;
    bool complete;
} ClusterArgs;


typedef struct ClusterConfigs {
    int ClusterNumber;
    int HashTable_number;
    int HashFunction_Number;
    int MAX_mHypercube;    
    int hypercube_dimensions;    
    int ProbesNumber;      
} ClusterConfigs;


void ClusterUsage(const char *);

void ClusterParse(int , char * const *, ClusterArgs *);

void ClusterConf(std::string, ClusterConfigs *);

void NormalizeDist(std::vector<float> &);

size_t BinarySearch(const std::vector<std::pair<float, size_t>> &, float);

float find_max(const std::vector<float> &);

bool CompareF(const std::pair<float, size_t> &, const std::pair<float, size_t> &);

bool in(const std::vector<size_t> &, size_t);

#endif // CLUSTER_UTILS_H
