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
    
private:
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
    
    class Symmetric2DDistribution{};
};
