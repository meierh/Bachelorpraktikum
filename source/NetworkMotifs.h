#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include "CommunicationPatterns.h"
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
    static std::vector<long double> networkTripleMotifs
    (
        const DistributedGraph& graph,
        unsigned int resultToRank = 0
    );
    /*-------------------------NetworkMotifs----------------------------------|||*/
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
        
        void setMotifTypes(std::vector<int> motifTypes)
        {
            for(int motifType : motifTypes)
            {
                assert(motifType>=1 && motifType<14);
                motifTypeBitArray |= (1<<motifType);
            }
        }
        void unsetMotifTypes(std::vector<int> motifTypes)
        {
            for(int motifType : motifTypes)
            {
                assert(motifType>=1 && motifType<14);
                motifTypeBitArray &= ~(1<<motifType);
            }
        }
        void unsetAllButMotifTypes(std::vector<int> motifTypes)
        {
            std::unordered_set<int> maintainedMotifTypes;
            maintainedMotifTypes.insert(motifTypes.begin(),motifTypes.end());
            std::vector<int> motifTypesToUnset;
            for(int motifType=1;motifType<14;motifType++)
            {
                if(maintainedMotifTypes.find(motifType)==maintainedMotifTypes.end())
                {
                    motifTypesToUnset.push_back(motifType);
                }
            }
            unsetMotifTypes(motifTypesToUnset);
        }
        bool isMotifTypeSet(int motifType)
        {
            assert(motifType>=1 && motifType<14);
            return motifTypeBitArray & (1<<motifType);
        }
        void printOut()
        {
            std::cout<<"-------------";
            for(int i=1;i<14;i++)
                if(isMotifTypeSet(i))
                    std::cout<<"1"<<" ";
                else
                    std::cout<<"0"<<" ";
            std::cout<<"--------------------"<<std::endl;
        }
        bool checkValidity()
        {
            bool res = motifTypeBitArray && !(motifTypeBitArray & (motifTypeBitArray-1));
            if(!res)
            {
                std::cout<<"---------------------";
                for(int i=1;i<13;i++)
                    if(isMotifTypeSet(i))
                        std::cout<<"1"<<" ";
                    else
                        std::cout<<"0"<<" ";
                std::cout<<"--------------------"<<res<<std::endl;
            }
            return res;
        }
    } threeMotifStructure;
};
