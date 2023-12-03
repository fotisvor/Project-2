#include <iostream>
#include <string>
#include <cstdint>
#include <chrono>
#include <sys/time.h>

#include "../../include/io_utils/cmd_args.h"
#include "../../include/io_utils/io_utils.h"
#include "../../include/modules/lsh/lsh.h"
#include "../../include/modules/exact_nn/exact_nn.h"
#include "../../include/metric/metric.h"

using namespace std;
#define C 1.2

void visualizeGraph(const std::vector<std::vector<std::pair<unsigned int, long unsigned int>>>& knn_graph, const std::string& filename) {
    // Open the file for writing
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    outFile << "Visualization of k-NN graph:" << std::endl;

    for (size_t i = 0; i < knn_graph.size(); ++i) {
        outFile << "Node " << i << " has distance " << knn_graph[i][0].second << " to neighbors: ";

        // Print neighbor indices
        for (const auto& neighbor : knn_graph[i]) {
            outFile << neighbor.first << " ";
        }

        outFile << std::endl;
    }

    outFile << "Graph visualization complete." << std::endl;

    // Close the file
    outFile.close();
}

// Function to visualize the results of gnnsSearch
void visualizeGnnsResultsToFile(const std::vector<std::pair<uint32_t, size_t>>& gnns_results, const std::string& output_file) {
    std::ofstream output_stream(output_file, std::ios::app); // Use append mode to avoid overwriting previous results

    if (!output_stream.is_open()) {
        std::cerr << "Error opening output file: " << output_file << std::endl;
        return;
    }

    output_stream << "GNNS Search Results:" << std::endl;
    for (const auto& result : gnns_results) {
        output_stream << "Node " << result.first << " has distance " << result.second << std::endl;
    }
    output_stream << "GNNS Search visualization complete." << std::endl;

    output_stream.close();
}

std::vector<std::pair<uint32_t, size_t>> gnnsSearch(
    const std::vector<std::vector<std::pair<uint32_t, size_t>>>& knn_graph,
    const std::vector<uint8_t>& query,
    const std::vector<std::vector<uint8_t>>& dataset) {

    // Convert query to MNISTImage
    MNISTImage queryImage;
    queryImage.features = std::vector<double>(query.begin(), query.end());

    // Convert dataset to vector of MNISTImage
    std::vector<MNISTImage> mnistDataset;
    for (const auto& data : dataset) {
        MNISTImage mnistImage;
        mnistImage.features = std::vector<double>(data.begin(), data.end());
        mnistDataset.push_back(mnistImage);
    }

    // Constants for GNNS algorithm
    const size_t R = 5;  // Number of random restarts
    const size_t T = 10; // Number of greedy steps
    const size_t L = 5;  // Number of points to return
    // Random restarts loop
    std::vector<std::pair<uint32_t, size_t>> best_results;
    for (size_t r = 0; r < R; ++r) {
        // Initialize Y0 as a random point uniformly over D
        size_t current_point_index = rand() % knn_graph.size();

        // Greedy steps loop
        for (size_t t = 0; t < T; ++t) {
            // Add neighbors of the current point to the set S
            const std::vector<std::pair<uint32_t, size_t>>& neighbors = knn_graph[current_point_index];
            if (neighbors.empty()) {
                std::cerr << "Error: No neighbors for point " << current_point_index << std::endl;
                
                break;
            }
            
            std::pair<uint32_t, size_t> closest_neighbor = neighbors[1];  // Assume the first neighbor is the closest initially
            
            for (const auto& neighbor : neighbors) {
                // Use calculateDistance to compute distances to the query
                
                double distance_neighbor = calculateDistanceL2(mnistDataset[neighbor.second], queryImage);
                

                // Check if the current neighbor is closer than the previously assumed closest neighbor
                if (distance_neighbor < calculateDistanceL2(mnistDataset[closest_neighbor.second], queryImage)) {
                    closest_neighbor = neighbor;
                }
            }
            
            // Update the current point to the one minimizing the distance
            current_point_index = closest_neighbor.second;
            
        }
        
        // Collect distances from the last greedy step
        best_results.push_back(knn_graph[current_point_index][0]); // Assuming you want to use the first neighbor in the vector
    }
    
    // Sort the distances in S and return the top L points with the smallest distances
    std::sort(best_results.begin(), best_results.end(), [&](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // Return the top L points
    return {best_results.begin(), best_results.begin() + std::min(L, best_results.size())};
}

static void LSH_Simulation(LSH_Args *args) 
{
     // Create LSH structure
    const uint16_t L = args->get_HashTableNum();
    const uint16_t N = args->get_NearNeighborNum();
    const uint32_t K = args->get_K();

    std::vector<std::vector<uint8_t>> dataset;
    read_file<uint8_t>(args->get_InputPath(), dataset);
    const double r = NN_distance<uint8_t>(dataset);

    LSH<uint8_t> lsh(L, N, K, r, dataset);
    
    std::vector<std::vector<std::pair<uint32_t, size_t>>> knn_graph(dataset.size());
    // Create the knn graph
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        knn_graph[i] = lsh.approximate_KNN(dataset[i]);
        auto stop = std::chrono::high_resolution_clock::now();
    }
    visualizeGraph(knn_graph,args->get_OutputPath());

    while (true) {
        std::vector<std::vector<uint8_t>> queries;
        read_file<uint8_t>(args->get_QueryPath(), queries);
        const size_t num_queries = queries.size();
        std::vector<std::vector<std::pair<uint32_t, size_t>>> ann_results(num_queries, std::vector<std::pair<uint32_t, size_t>>(N));
        std::vector<std::chrono::microseconds> ann_query_times(num_queries);
        std::vector<std::chrono::microseconds> enn_query_times(num_queries);
        std::vector<std::vector<std::pair<uint32_t, size_t>>> gnns_results(num_queries);
        
        for (size_t i = 0; i < num_queries; ++i) {
            const auto& query = queries[i];
            auto start = std::chrono::high_resolution_clock::now();
            // Perform GNNS search using the k-NN graph
            gnns_results[i] = gnnsSearch(knn_graph, query,dataset);


        }
        visualizeGnnsResultsToFile(gnns_results[0], args->get_OutputPath());
        //visualizeGnnsResults(gnns_results);
        //print_statistics(N, num_queries, ann_results, ann_query_times, enn_distances, enn_query_times);

        //write_output(args->get_output_file_path(), N, num_queries, ann_results, ann_query_times, enn_distances, enn_query_times, range_results, "LSH");

        std::cout << "You can now open the output file and see its contents" << std::endl;

        if (user_prompt_file("\nDo you want to continue the simulation and repeat the search process?: [Y/N]: ") != "Y") {
            break;
        }

        if (user_prompt_file("\nDo you want to use the same query file?: [Y/N]: ") != "Y") {
            args->set_QueryPath(user_prompt_file("\nEnter the path to the new query file: "));
        }
    }
}

int main(int argc, char *argv[]) 
{
    LSH_Args *args = nullptr;
    if (argc==19) 
    {
        LSH_parse(argc, argv, &args);
    } else if (argc == 1) 
    {
        user_interface(&args);
    } 
    else 
    {
        LSH_Usage(argv[0]);
        return EXIT_FAILURE;
    }
    LSH_Simulation(args);
    
    delete args;
    return EXIT_SUCCESS;
}
