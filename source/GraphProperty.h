#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include <numeric>

class GraphProperty {
public:
    
    /* Foreach combination of areas A and B the function sums the strength
     * of all edges connecting a node in area A with a node in area B.
     * 
     * Parameter: A DistributedGraph (Function is MPI compliant)
     * Return: OPEN  
     */
    struct stdPair_hash
    {template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const 
        {return std::hash<T1>{}(p.first) ^  std::hash<T2>{}(p.second);}
    };
    using AreaConnecMap = std::unordered_map<std::pair<std::string,std::string>,int,stdPair_hash>;
    static std::unique_ptr<AreaConnecMap> areaConnectivityStrength(const DistributedGraph& graph,int resultToRank=0);
};
