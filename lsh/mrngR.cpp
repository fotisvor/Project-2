#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <numeric>
#include <limits>
#include <random>
#include <queue>
#include <chrono>
#include <string>

// Define a structure to represent an MNIST image
struct MNISTImage {
    std::vector<double> features; // Vector to store pixel values as features
};

// Function to calculate Euclidean distance between two MNIST images
double calculateDistance(const MNISTImage& img1, const MNISTImage& img2) {
    double sum = 0.0;
    // Iterate over each feature (pixel) and calculate the squared difference
    for (size_t i = 0; i < img1.features.size(); ++i) {
        double diff = img1.features[i] - img2.features[i];
        sum += diff * diff;
    }
    // Return the square root of the sum to get the Euclidean distance
    return std::sqrt(sum);
}

// Structure to store search results, including indices, true distances, and approximate distances
struct SearchResults {
    std::vector<size_t> indices;
    std::vector<double> trueDistances;
    std::vector<double> approximateDistances;
};

// Function to approximate the distance between two MNIST images based on MRNG construction
double approximateDistanceFunction(const MNISTImage& img1, const MNISTImage& img2, const std::set<size_t>& neighbors) {
    // Implement your logic to approximate the distance based on MRNG construction
    // For example, you can compare distances with the neighbors of img1
    // Return a value indicating the approximate distance
    // This can be a simple comparison or a more complex function based on your MRNG properties
    return calculateDistance(img1, img2);
}

// Function to perform a search on the graph with distances and return results
SearchResults searchOnGraphWithDistances(const std::vector<std::set<size_t>>& graph, size_t start,
                                         const MNISTImage& query, size_t k,
                                         const std::vector<MNISTImage>& mnistDataset, size_t candidateLimit) {
    SearchResults result;
    std::vector<bool> checked(graph.size(), false);
    std::vector<std::pair<size_t, double>> candidates;

    size_t i = 0;
    result.indices.push_back(start);
    checked[start] = true;

    while (i < k) {
        size_t current = result.indices.back();

        // Iterate over neighbors of the current node
        for (size_t neighbor : graph[current]) {
            if (!checked[neighbor]) {
                double trueDistance = calculateDistance(query, mnistDataset[neighbor]);
                double approximateDistance = approximateDistanceFunction(mnistDataset[current], mnistDataset[neighbor], graph[current]);

                candidates.emplace_back(neighbor, trueDistance);

                // Store both true and approximate distances
                result.trueDistances.push_back(trueDistance);
                result.approximateDistances.push_back(approximateDistance);
            }
        }

        // Sort candidates in ascending order of distance to q
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.second < rhs.second;
                  });

        // Add the closest node to the result
        result.indices.push_back(candidates[0].first);
        checked[candidates[0].first] = true;
        candidates.clear();

        ++i;
    }

    return result;
}

// Function to construct the MRNG (Multiple Random Neighbor Graph)
std::vector<std::set<size_t>> constructMRNG(const std::vector<MNISTImage>& dataset, size_t candidateLimit) {
    std::vector<std::set<size_t>> neighbors(dataset.size());

    // Initialize a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    auto startTimeGraphConstruction = std::chrono::high_resolution_clock::now();

    for (size_t p = 0; p < dataset.size(); ++p) {
        std::vector<size_t> candidateNeighbors(candidateLimit);
        std::iota(candidateNeighbors.begin(), candidateNeighbors.end(), 0);

        // Shuffle the candidate neighbors randomly
        std::shuffle(candidateNeighbors.begin(), candidateNeighbors.end(), gen);

        for (size_t q : candidateNeighbors) {
            if (p == q) continue;

            bool isValidEdge = true;

            for (size_t r : neighbors[p]) {
                if (calculateDistance(dataset[p], dataset[q]) > calculateDistance(dataset[r], dataset[q])) {
                    isValidEdge = false;
                    break;
                }
            }

            if (isValidEdge) {
                neighbors[p].insert(q);
            }
        }
    }

    auto endTimeGraphConstruction = std::chrono::high_resolution_clock::now();
    auto elapsedTimeGraphConstruction = std::chrono::duration_cast<std::chrono::seconds>(endTimeGraphConstruction - startTimeGraphConstruction).count();

    std::cout << "Graph Construction Time: " << elapsedTimeGraphConstruction << " seconds" << std::endl;

    return neighbors;
}

// Function to find the nearest neighbor on the MRNG
size_t findNearestNeighborOnGraph(const std::vector<MNISTImage>& dataset, const MNISTImage& query,
                                  const std::vector<std::set<size_t>>& mrngEdges,
                                  size_t candidateLimit) {
    size_t startNode = 0; // You can choose any starting node here
    return searchOnGraphWithDistances(mrngEdges, startNode, query, 1, dataset, candidateLimit).indices[1];
}

// Function to find the top N neighbors on the MRNG
std::vector<size_t> findTopNNeighborsOnGraph(const std::vector<MNISTImage>& dataset, const MNISTImage& query,
                                             const std::vector<std::set<size_t>>& mrngEdges, size_t N,
                                             size_t candidateLimit) {
    return searchOnGraphWithDistances(mrngEdges, findNearestNeighborOnGraph(dataset, query, mrngEdges, candidateLimit), query, N, dataset, candidateLimit).indices;
}

// Function to load MNIST images from a binary file
std::vector<MNISTImage> loadMNISTImages(const std::string& filename, int numImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Skip header information
    file.seekg(16);

    std::vector<MNISTImage> dataset(numImages);
    for (int i = 0; numImages > 0 && i < numImages; ++i) {
        dataset[i].features.resize(28 * 28);

        for (int j = 0; j < 28 * 28; ++j) {
            uint8_t pixelValue;
            file.read(reinterpret_cast<char*>(&pixelValue), sizeof(pixelValue));
            dataset[i].features[j] = static_cast<double>(pixelValue) / 255.0;
        }
    }

    return dataset;
}

// Function to visualize the MRNG using Graphviz DOT format
void visualizeMRNG(const std::vector<std::set<size_t>>& neighbors, const std::string& filename) {
    std::ofstream dotFile(filename);

    if (!dotFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    dotFile << "graph MRNG {" << std::endl;

    for (size_t p = 0; p < neighbors.size(); ++p) {
        for (size_t r : neighbors[p]) {
            if (p < r) {
                dotFile << "  " << p << " -- " << r << ";" << std::endl;
            }
        }
    }

    dotFile << "}" << std::endl;
    dotFile.close();
}

int main(int argc, char* argv[]) {
    const int numImages = 10000;
    printf("numImages: %d\n", numImages);
    // Check command line arguments
    if (argc != 6) {
        std::cerr << "Usage: " << argc << " -d input_file -q query_file -N <numNeighbors> -l <candidateLimit> -o output_file" << std::endl;
        return EXIT_FAILURE;
    }
    std::string inputFilename;
    std::string queryFilename;
    std::string outputFilename;
    size_t candidateLimit;
    size_t numNeighbors;

        
    printf("N: %s\n", argv[3]);
    numNeighbors = std::stoi(argv[3]);
        
            printf("l: %s\n", argv[4]);
            candidateLimit = std::stoi(argv[4]);
        
            printf("d: %s\n", argv[1]);
            inputFilename = argv[1];
        
            printf("q: %s\n", argv[2]);
            queryFilename = argv[2];
        
            printf("o: %s\n", argv[5]);
            outputFilename = argv[5];
        

    printf("inputFilename: %s\n", inputFilename.c_str());
    // Load MNIST dataset
    std::vector<MNISTImage> mnistDataset = loadMNISTImages(inputFilename, numImages);

    auto startTimeTotal = std::chrono::high_resolution_clock::now();

    // Construct MRNG graph
    std::vector<std::set<size_t>> mrngEdges = constructMRNG(mnistDataset, candidateLimit);

    // Load query image
    MNISTImage queryImage;
    queryImage.features.resize(28 * 28);
    std::ifstream queryFile(queryFilename);
    if (!queryFile.is_open()) {
        std::cerr << "Error opening query file." << std::endl;
        return EXIT_FAILURE;
    }
    printf("queryFilename: %s\n", queryFilename.c_str());
    for (int i = 0; i < 28 * 28; ++i) {
        uint8_t pixelValue;
        queryFile.read(reinterpret_cast<char*>(&pixelValue), sizeof(pixelValue));
        queryImage.features[i] = static_cast<double>(pixelValue) / 255.0;
    }

    // Find nearest neighbor on the MRNG
    size_t nearestNeighborOnGraph = findNearestNeighborOnGraph(mnistDataset, queryImage, mrngEdges, candidateLimit);
    double nearestNeighborDistanceOnGraph = calculateDistance(queryImage, mnistDataset[nearestNeighborOnGraph]);

    std::cout << "Nearest Neighbor Index (Search on Graph): " << nearestNeighborOnGraph
              << ", Distance: " << nearestNeighborDistanceOnGraph << std::endl;

    auto startTimeSearch = std::chrono::high_resolution_clock::now();

    // Perform search on the MRNG
    SearchResults searchResults = searchOnGraphWithDistances(mrngEdges, nearestNeighborOnGraph, queryImage, numNeighbors, mnistDataset, candidateLimit);

    auto endTimeSearch = std::chrono::high_resolution_clock::now();
    auto elapsedTimeSearch = std::chrono::duration_cast<std::chrono::seconds>(endTimeSearch - startTimeSearch).count();

    // Print true and approximate distances
    std::cout << "Search Execution Time: " << elapsedTimeSearch << " seconds" << std::endl;
    std::cout << "Top " << numNeighbors << " Neighbors (Search on Graph):" << std::endl;

    for (size_t i = 0; i < numNeighbors && i < searchResults.indices.size(); ++i) {
        size_t neighborIndex = searchResults.indices[i];
        double trueDistance = searchResults.trueDistances[i];
        double approximateDistance = searchResults.approximateDistances[i];

        std::cout << "Index: " << neighborIndex
                  << ", True Distance: " << trueDistance
                  << ", Approximate Distance: " << approximateDistance << std::endl;
    }

    // Visualize the MRNG graph
    visualizeMRNG(mrngEdges, "mrng_visualization.dot");

    auto endTimeTotal = std::chrono::high_resolution_clock::now();
    auto elapsedTimeTotal = std::chrono::duration_cast<std::chrono::seconds>(endTimeTotal - startTimeTotal).count();

    std::cout << "Total Execution Time: " << elapsedTimeTotal << " seconds" << std::endl;

    return 0;
}
