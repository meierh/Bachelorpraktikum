#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include "CommunicationPatterns.h"
#include "NodeCounter.h"
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <stdexcept>
#include <cassert>
#include <algorithm>

class NetworkMotifs {
public:
    /*|||-----------------------NetworkMotifs--------------------------------
     *
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     * 
     */
    static std::array<long double,14> compute_network_TripleMotifs
    (
        const DistributedGraph& graph,
        unsigned int resultToRank = 0
    );
    /*-------------------------NetworkMotifs----------------------------------|||*/

    static std::array<long double,14> compute_network_TripleMotifs_SingleProc
    (
    const DistributedGraph& graph,
    unsigned int my_rank = 0
    );

private:
    struct StdPair_hash
    {
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const 
        {
            return std::hash<T1>{}(p.first) ^  std::hash<T2>{}(p.second);
        }
    };  
    
    typedef struct
    {
        std::uint64_t node_1_rank;
        std::uint64_t node_1_local;
        std::uint64_t node_2_rank;
        std::uint64_t node_2_local;
        std::uint64_t node_3_rank;
        std::uint64_t node_3_local;
        std::uint64_t motifTypeBitArray=0;
        
        void selfTest();        
        void setMotifTypes(std::vector<int> motifTypes);
        void unsetMotifTypes(std::vector<int> motifTypes);
        void unsetAllButMotifTypes(std::vector<int> motifTypes);
        bool isMotifTypeSet(int motifType);
        void printOut();
        bool checkValidity();
    } threeMotifStructure;

    static std::uint16_t update_edge_bitArray
    (
        const DistributedGraph& graph,
        std::uint16_t exists_edge_bitArray,
        unsigned int node_2_rank, 
        std::uint64_t node_2_local, 
        unsigned int node_3_rank, 
        std::uint64_t node_3_local
    );
};
