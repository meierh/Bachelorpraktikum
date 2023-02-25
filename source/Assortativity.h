#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <cassert>
#include "CommunicationPatterns.h"
#include <numeric>

class Assortativity {
public:
    std::tuple<double,double,double,double> compute_assortativity_coefficient
    (
        const DistributedGraph& graph
    );
    
private:
    std::pair<int,int> compute_max_nodeDegree_OutIn
    (
        const DistributedGraph& graph
    );
    
    std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> compute_node_degree_distribution
    (
        const DistributedGraph& graph
    );
    
    /* Assortative mixing in networks by M. E. J. Newman
     * 
     */
    std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> compute_normalized_node_degree_distribution
    (
        const DistributedGraph& graph
    );
    
    std::pair<double,double> compute_standard_deviation_of_node_degree_distribution
    (
        const std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>>& node_degree_distribution
    );
    
    typedef struct
    {
        std::uint64_t source_OutDegree;
        std::uint64_t source_InDegree;
        std::uint64_t target_OutDegree;
        std::uint64_t target_InDegree;
    } EdgeDegrees;    
    
    class Distribution2D{
        public:
            Distribution2D(int first_dimension, int second_dimension);
            Distribution2D();
            
            void set_probability(int first_index, int second_index, double probability);
            double get_probability(int first_index, int second_index);
            int get_first_dimension(){return first_dimension;};
            int get_second_dimension(){return second_dimension;};
            void reset(int first_dimension, int second_dimension);
        
        private:
            int first_dimension;
            int second_dimension;
            std::vector<std::vector<double>> probabilities;
    };
    
    std::unique_ptr<std::tuple<Distribution2D,Distribution2D,Distribution2D,Distribution2D>> compute_joint_edge_degree_distribution
    (
        const DistributedGraph& graph
    );
};
