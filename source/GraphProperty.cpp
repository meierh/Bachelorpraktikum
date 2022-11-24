#include "GraphProperty.h"

std::unique_ptr<GraphProperty::AreaConnecMap> GraphProperty::areaConnectivityStrength
(
    const DistributedGraph& graph,
    int resultToRank
)
{
    AreaConnecMap areaConnectivityStrengthMapLocal;
    const int my_rank = MPIWrapper::get_my_rank();
    std::unordered_multimap<int,std::pair<int,int>> rankToNodeAndArea
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        const std::vector<InEdge>& iEdges = graph.get_out_edges(my_rank,node_local_ind);
        

    }
    
    std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        const std::string source_area_name = graph.get_node_area_name(my_rank,node_local_ind);
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_local_ind);
        for(const OutEdge& oEdge : oEdges)
        {            
            //Main communication effort
            const std::string target_area_name = graph.get_node_area_name(oEdge.target_rank,oEdge.target_id);

            std::pair<std::string,std::string> area_to_area(source_area_name,target_area_name);            
            // Non existend key has the value 0 because of value initialization
            // https://en.cppreference.com/w/cpp/container/unordered_map/operator_at
            areaConnectivityStrengthMapLocal[area_to_area]+=oEdge.weight;
        }
    }
    
    std::vector<std::string> sourceArea;
    std::vector<int> sourceAreaStringLen;
    std::vector<std::string> targetArea;
    std::vector<int> targetAreaStringLen;
    std::vector<int> weightSum;
    int nbrOfEntries = sourceArea.size();
    
    for(auto keyValue = areaConnectivityStrengthMapLocal.begin();
        keyValue != areaConnectivityStrengthMapLocal.end();
        ++keyValue)
    {
        auto& sourceArea_targetArea = keyValue->first;        
        sourceArea.push_back(sourceArea_targetArea.first);
        sourceAreaStringLen.push_back(sourceArea_targetArea.first.size());
        targetArea.push_back(sourceArea_targetArea.second);
        targetAreaStringLen.push_back(sourceArea_targetArea.second.size());
        weightSum.push_back(keyValue->second);
    }

    std::unique_ptr<AreaConnecMap> result;
    if(my_rank==resultToRank)
    {
        result = std::make_unique<AreaConnecMap>();
        for(int otherRank=0; otherRank<MPIWrapper::get_number_ranks(); otherRank++)
        {
            if(otherRank==resultToRank)
                continue;

            MPI_Status status;
            
            int otherRankNbrEntries;
            MPI_Recv(&otherRankNbrEntries,1,MPI_INT,otherRank,0,MPI_COMM_WORLD,&status);
            
            std::vector<int> otherRankSourceAreaStringLen(otherRankNbrEntries);
            std::vector<int> otherRankTargetAreaStringLen(otherRankNbrEntries);
            MPI_Recv(otherRankSourceAreaStringLen.data(),otherRankNbrEntries,MPI_INT,otherRank,1,MPI_COMM_WORLD,&status);
            MPI_Recv(otherRankTargetAreaStringLen.data(),otherRankNbrEntries,MPI_INT,otherRank,2,MPI_COMM_WORLD,&status);
            
            std::vector<std::string> otherRankSourceArea(otherRankNbrEntries);
            std::vector<std::string> otherRankTargetArea(otherRankNbrEntries);
            int charArraySize =  std::reduce(otherRankSourceAreaStringLen.begin(),otherRankSourceAreaStringLen.end())
                                +std::reduce(otherRankTargetAreaStringLen.begin(),otherRankTargetAreaStringLen.end());
            char* AreaCharArray = new char[charArraySize];
            MPI_Recv(AreaCharArray,charArraySize,MPI_CHAR,otherRank,3,MPI_COMM_WORLD,&status);
            int pChar=0;
            for(int i=0;i<otherRankSourceAreaStringLen.size();i++)
            {
                otherRankSourceArea[i] = std::string(&(AreaCharArray[pChar]),otherRankSourceAreaStringLen[i]);
                pChar+=otherRankSourceAreaStringLen[i];
            }
            for(int i=0;i<otherRankTargetAreaStringLen.size();i++)
            {
                otherRankTargetArea[i] = std::string(&(AreaCharArray[pChar]),otherRankTargetAreaStringLen[i]);
                pChar+=otherRankTargetAreaStringLen[i];
            }
            delete[] AreaCharArray;
            
            std::vector<int> otherRankWeightSum(otherRankNbrEntries);
            MPI_Recv(otherRankWeightSum.data(),otherRankNbrEntries,MPI_INT,otherRank,4,MPI_COMM_WORLD,&status);

            for(int entryI=0;entryI<otherRankNbrEntries;entryI++)
            {
                std::pair<std::string,std::string> area_to_area(otherRankSourceArea[entryI],otherRankTargetArea[entryI]);
                // Non existend key has the value 0 because of value initialization
                // https://en.cppreference.com/w/cpp/container/unordered_map/operator_at
                (*result)[area_to_area]+=otherRankWeightSum[entryI];
            }
        }
        for(int entryI=0;entryI<nbrOfEntries;entryI++)
        {
            std::pair<std::string,std::string> area_to_area(sourceArea[entryI],targetArea[entryI]);
            // Non existend key has the value 0 because of value initialization
            // https://en.cppreference.com/w/cpp/container/unordered_map/operator_at
            (*result)[area_to_area]+=weightSum[entryI];
        }
    }
    else
    {
        MPI_Send(&nbrOfEntries,1,MPI_INT,resultToRank,0,MPI_COMM_WORLD);
        
        MPI_Send(sourceAreaStringLen.data(),nbrOfEntries,MPI_INT,resultToRank,1,MPI_COMM_WORLD);
        MPI_Send(targetAreaStringLen.data(),nbrOfEntries,MPI_INT,resultToRank,2,MPI_COMM_WORLD);
        
        int charArraySize =  std::reduce(sourceAreaStringLen.begin(),sourceAreaStringLen.end())
                            +std::reduce(targetAreaStringLen.begin(),targetAreaStringLen.end());
        char* AreaCharArray = new char[charArraySize];
        int pChar=0;
        for(int i=0;i<sourceArea.size();i++)
            for(int j=0;j<sourceArea[i].size();j++)
                AreaCharArray[pChar++] = sourceArea[i].c_str()[j];
        for(int i=0;i<targetArea.size();i++)
            for(int j=0;j<targetArea[i].size();j++)
                AreaCharArray[pChar++] = targetArea[i].c_str()[j];
        MPI_Send(AreaCharArray,charArraySize,MPI_CHAR,resultToRank,3,MPI_COMM_WORLD);
        delete[] AreaCharArray;
        
        MPI_Send(weightSum.data(),nbrOfEntries,MPI_INT,resultToRank,4,MPI_COMM_WORLD);
    }
    
    return std::move(result);
}
