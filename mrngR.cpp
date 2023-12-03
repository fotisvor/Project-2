#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <vector>

// Define a structure to represent an MNIST image
struct MNISTImage
{
    std::vector<double> features; // Vector to store pixel values as features
};

// Structure to store search results
struct SearchResults
{
    std::vector<size_t> indices;
    std::vector<double> trueDistances;
    std::vector<double> approximateDistances;
    std::vector<double> approximationFactors;
};

// Calculate Euclidean distance between two MNIST images
double calculateDistance(const MNISTImage &img1, const MNISTImage &img2)
{
    double sum = 0.0;
    // Iterate over each feature (pixel) and calculate the squared difference
    for (size_t i = 0; i < img1.features.size(); ++i)
    {
        double diff = img1.features[i] - img2.features[i];
        sum += diff * diff;
    }
    // Return the square root for Euclidean distance
    return std::sqrt(sum);
}

// Function to approximate the distance between two MNIST images based on MRNG construction
double approximateDistanceFunction(const MNISTImage &img1, const MNISTImage &img2, const std::set<size_t> &neighbors)
{
    // Maybe better implementation here
    return calculateDistance(img1, img2);
}

// Function to perform a search on the graph with distances and return results
SearchResults searchOnGraphWithDistances(const std::vector<std::set<size_t>> &graph, size_t start,
                                         const MNISTImage &query, size_t k, const std::vector<MNISTImage> &mnistDataset,
                                         size_t candidateLimit)
{
    SearchResults result;
    std::vector<bool> checked(graph.size(), false);
    std::vector<std::pair<size_t, double>> candidates;

    size_t i = 0;
    result.indices.push_back(start);
    checked[start] = true;

    while (i < k)
    {
        size_t current = result.indices.back();

        // Iterate over neighbors of the current node
        for (size_t neighbor : graph[current])
        {
            if (!checked[neighbor])
            {
                double trueDistance = calculateDistance(query, mnistDataset[neighbor]);
                double approximateDistance =
                    approximateDistanceFunction(mnistDataset[current], mnistDataset[neighbor], graph[current]);

                candidates.emplace_back(neighbor, trueDistance);

                // Store both true and approximate distances
                result.trueDistances.push_back(trueDistance);
                result.approximateDistances.push_back(approximateDistance);

                // Calculate and store the approximation factor
                double approximationFactor = approximateDistance / trueDistance;
                result.approximationFactors.push_back(approximationFactor);
            }
        }

        // Sort candidates in ascending order of distance to q
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });

        // Add the closest node to the result
        result.indices.push_back(candidates[0].first);
        checked[candidates[0].first] = true;
        candidates.clear();

        ++i;
    }

    return result;
}

// Function to construct the MRNG
std::vector<std::set<size_t>> constructMRNG(const std::vector<MNISTImage> &dataset, size_t candidateLimit)
{
    std::vector<std::set<size_t>> neighbors(dataset.size());

    // Initialize a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    auto startTimeGraphConstruction = std::chrono::high_resolution_clock::now();

    for (size_t p = 0; p < dataset.size(); ++p)
    {
        std::vector<size_t> candidateNeighbors(candidateLimit);
        std::iota(candidateNeighbors.begin(), candidateNeighbors.end(), 0);

        // Shuffle the candidate neighbors randomly
        std::shuffle(candidateNeighbors.begin(), candidateNeighbors.end(), gen);

        for (size_t q : candidateNeighbors)
        {
            if (p == q)
                continue;

            bool isValidEdge = true;

            for (size_t r : neighbors[p])
            {
                if (calculateDistance(dataset[p], dataset[q]) > calculateDistance(dataset[r], dataset[q]))
                {
                    isValidEdge = false;
                    break;
                }
            }

            if (isValidEdge)
            {
                neighbors[p].insert(q);
            }
        }
    }

    auto endTimeGraphConstruction = std::chrono::high_resolution_clock::now();
    auto elapsedTimeGraphConstruction =
        std::chrono::duration_cast<std::chrono::seconds>(endTimeGraphConstruction - startTimeGraphConstruction).count();

    std::cout << "Graph Construction Time: " << elapsedTimeGraphConstruction << " seconds" << std::endl;

    return neighbors;
}

// Function to find the nearest neighbor on the MRNG
size_t findNearestNeighborOnGraph(const std::vector<MNISTImage> &dataset, const MNISTImage &query,
                                  const std::vector<std::set<size_t>> &mrngEdges, size_t candidateLimit)
{
    size_t startNode = 0; // You can choose any starting node here
    return searchOnGraphWithDistances(mrngEdges, startNode, query, 1, dataset, candidateLimit).indices[1];
}

// Function to find the top/closest N neighbors on the MRNG
std::vector<size_t> findTopNNeighborsOnGraph(const std::vector<MNISTImage> &dataset, const MNISTImage &query, const std::vector<std::set<size_t>> &mrngEdges, size_t N,
                                             size_t candidateLimit)
{
    return searchOnGraphWithDistances(mrngEdges, findNearestNeighborOnGraph(dataset, query, mrngEdges, candidateLimit), query, N, dataset, candidateLimit)
        .indices;
}

// Function to load MNIST images from a binary file
std::vector<MNISTImage> loadMNISTImages(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get the size of the file
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate the number of images based on file size
    size_t imageSize = 28 * 28 + 1; // Each image is 28*28 pixels, plus 1 for the label
    size_t numImages = static_cast<size_t>(fileSize / imageSize);

    // Print the number of images being read
    std::cout << "Reading " << numImages << " images from the input file." << std::endl;

    // Skip header information
    file.seekg(16);

    std::vector<MNISTImage> dataset(numImages);
    for (size_t i = 0; i < numImages; ++i)
    {
        dataset[i].features.resize(28 * 28);

        for (int j = 0; j< 28 * 28; ++j)
        {
            uint8_t pixelValue;
            file.read(reinterpret_cast<char *>(&pixelValue), sizeof(pixelValue));
            dataset[i].features[j] = static_cast<double>(pixelValue) / 255.0;
        }
    }

    return dataset;
}

// Function to visualize the MRNG
void visualizeMRNG(const std::vector<std::set<size_t>> &neighbors, const std::string &filename)
{
    std::ofstream dotFile(filename);

    if (!dotFile.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    dotFile << "graph MRNG {" << std::endl;

    for (size_t p = 0; p < neighbors.size(); ++p)
    {
        for (size_t r : neighbors[p])
        {
            if (p < r)
            {
                dotFile << "  " << p << " -- " << r << ";" << std::endl;
            }
        }
    }

    dotFile << "}" << std::endl;
    dotFile.close();
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " input_file query_file numNeighbors candidateLimit output_file"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::string inputFilename = argv[1];
    std::string queryFilename = argv[2];
    size_t numNeighbors = std::stoi(argv[3]);
    size_t candidateLimit = std::stoi(argv[4]);
    std::string outputFilename = argv[5];

    std::vector<MNISTImage> mnistDataset = loadMNISTImages(inputFilename);

    auto startTimeTotal = std::chrono::high_resolution_clock::now();

    std::vector<std::set<size_t>> mrngEdges = constructMRNG(mnistDataset, candidateLimit);

    std::ifstream queryFile(queryFilename, std::ios::binary);
    if (!queryFile.is_open())
    {
        std::cerr << "Error opening query file." << std::endl;
        return EXIT_FAILURE;
    }

    auto startTimeSearch = std::chrono::high_resolution_clock::now();

    std::ofstream outputFile(outputFilename);
    std::streambuf *originalStdout = std::cout.rdbuf();
    std::cout.rdbuf(outputFile.rdbuf());

    for (size_t queryIndex = 0; queryIndex < mnistDataset.size(); ++queryIndex)
    {
        MNISTImage queryImage;
        queryImage.features.resize(28 * 28);

        for (int i = 0; i < 28 * 28; ++i)
        {
            uint8_t pixelValue;
            queryFile.read(reinterpret_cast<char *>(&pixelValue), sizeof(pixelValue));
            queryImage.features[i] = static_cast<double>(pixelValue) / 255.0;
        }

        size_t nearestNeighborOnGraph = findNearestNeighborOnGraph(mnistDataset, queryImage, mrngEdges, candidateLimit);
        double nearestNeighborDistanceOnGraph = calculateDistance(queryImage, mnistDataset[nearestNeighborOnGraph]);

        std::cout << "Query " << queryIndex + 1
                  << ": Nearest Neighbor Index (Search on Graph): " << nearestNeighborOnGraph
                  << ", Distance: " << nearestNeighborDistanceOnGraph << std::endl;

        SearchResults searchResults = searchOnGraphWithDistances(mrngEdges, nearestNeighborOnGraph, queryImage, numNeighbors, mnistDataset, candidateLimit);

        std::cout << "Top " << numNeighbors << " Neighbors (Search on Graph):" << std::endl;
        for (size_t i = 0; i < numNeighbors && i < searchResults.indices.size(); ++i)
        {
            size_t neighborIndex = searchResults.indices[i];
            double trueDistance = searchResults.trueDistances[i];
            double approximateDistance = searchResults.approximateDistances[i];
            double approximationFactor = searchResults.approximationFactors[i];

            std::cout << "Index: " << neighborIndex << ", True Distance: " << trueDistance
                      << ", Approximate Distance: " << approximateDistance << ", Approximation Factor: " << approximationFactor << std::endl;
        }

        double maxApproximationFactor = *std::max_element(searchResults.approximationFactors.begin(), searchResults.approximationFactors.end());
        std::cout << "Maximum Approximation Factor: " << maxApproximationFactor << std::endl;
    }

    std::cout.rdbuf(originalStdout);

    auto endTimeSearch = std::chrono::high_resolution_clock::now();
    auto elapsedTimeSearch = std::chrono::duration_cast<std::chrono::seconds>(endTimeSearch - startTimeSearch).count();

    std::cout << "Search Execution Time: " << elapsedTimeSearch << " seconds" << std::endl;
    visualizeMRNG(mrngEdges, "mrng_visualization.dot");

    auto endTimeTotal = std::chrono::high_resolution_clock::now();
    auto elapsedTimeTotal = std::chrono::duration_cast<std::chrono::seconds>(endTimeTotal - startTimeTotal).count();
    std::cout << "Total Execution Time: " << elapsedTimeTotal << " seconds" << std::endl;

    return 0;
}

