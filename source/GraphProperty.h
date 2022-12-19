#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"

class GraphProperty {
public:
    
    /* Foreach combination of areas A and B the function sums the strength
     * of all edges connecting a node in area A with a node in area B.
     * 
     * Parameter: A DistributedGraph (Function is MPI compliant)
     * Return: OPEN  
     */
    static int areaConnectivityStrength(const DistributedGraph& graph);
};
