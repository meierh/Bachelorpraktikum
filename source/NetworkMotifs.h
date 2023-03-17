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
	 * Function to compute the fraction of networkMotifs of the graph
	 *
	 * Returns: The fraction of NetworkMotifs in regard to their total count using 
     *          the network motif types and numeration according to paper
	 *  "A tutorial in connectome analysis: Topological and spatial features of brain networks"
	 *  by Marcus Kaiser, 2011 in NeuroImage, (892-907), page 899
     *          result[0] is the total count of motifs
     *          result[i] is the fraction of motif i [1...13] (total count motif i / result[0])
	 *
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     * 
	 * MPI Constraints: Function must be called on all ranks simultaneously
	 * 					Function returns correct information to all ranks
     */
    static std::array<long double,14> compute_network_TripleMotifs
    (
        DistributedGraph& graph,
        unsigned int resultToRank = 0
    );
    /*-------------------------NetworkMotifs----------------------------------|||*/

    static std::array<long double,14> compute_network_TripleMotifs_SingleProc
    (
        DistributedGraph& graph,
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
        std::uint64_t node_3_rank;
        std::uint64_t node_3_local;
        std::uint16_t motifTypeBitArray=0;
        
        void selfTest();        
        void setMotifTypes(std::vector<int> motifTypes);
        void unsetMotifTypes(std::vector<int> motifTypes);
        void unsetAllButMotifTypes(std::vector<int> motifTypes);
        bool isMotifTypeSet(int motifType);
        void printOutComplete();
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
