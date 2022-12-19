#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include <numeric>
#include <unordered_map>
#include <cassert>  // debug

class GraphProperty {
public:
    
    /* Foreach combination of areas A and B the function sums the strength
     * of all edges connecting a node in area A with a node in area B.
     * 
     * Parameter: A DistributedGraph (Function is MPI compliant)
     * Return: OPEN  
     */
    struct stdPair_hash
    {
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const 
        {
            return std::hash<T1>{}(p.first) ^  std::hash<T2>{}(p.second);
        }
    };
    struct stdDoublePair_hash
    {
        stdPair_hash hash;
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const 
        {
            return hash(p.first) ^  hash(p.second);
        }
    };    
    using AreaConnecMap = std::unordered_map<std::pair<std::string,std::string>,int,stdPair_hash>;
    using AreaLocalID = std::pair<std::uint64_t,std::uint64_t>;
    using AreaIDConnecMap = std::unordered_map<std::pair<AreaLocalID,AreaLocalID>,int,stdDoublePair_hash>;
    static std::unique_ptr<AreaConnecMap> areaConnectivityStrength(const DistributedGraph& graph,int resultToRank=0);
    static std::unique_ptr<AreaConnecMap> areaConnectivityStrengthSingleProc(const DistributedGraph& graph,int resultToRank=0);
    
private:
    
    /* General method for transfering one date of the type DATA from any in edge to any out edge
     */
    template<typename DATA>
    using collectedData_ptr = std::unique_ptr<std::vector<std::vector<DATA>>>;
    using collectedDataStructure_ptr = std::unique_ptr<std::vector<std::unordered_map<std::uint64_t,int>>>;
    template<typename DATA>
    static std::pair<collectedData_ptr<DATA>,collectedDataStructure_ptr> collectAlongEdges_InToOut
    (const DistributedGraph& graph,MPI_Datatype datatype,std::function<DATA(int,int)> date_get_function);
};
