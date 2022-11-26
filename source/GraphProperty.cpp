#include "GraphProperty.h"

std::unique_ptr<GraphProperty::AreaConnecMap> GraphProperty::areaConnectivityStrength
(
    const DistributedGraph& graph,
    int resultToRank
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    int ownRankRecvInd;
    
    //prepare buffer for areaID data from other ranks
    std::unordered_map<int,std::pair<int,int>> treated_ranks_to_pair_ind_size_recv;
    std::vector<int> ranks_recv;
    std::vector<std::vector<std::uint64_t>> rank_ind_to_local_NodeID;
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_local_ind);
        for(const OutEdge& oEdge : oEdges)
        {
            const auto rank_ind = treated_ranks_to_pair_ind_size_recv.find(oEdge.target_rank);
            if(rank_ind != treated_ranks_to_pair_ind_size_recv.end())
            {
                rank_ind_to_local_NodeID[rank_ind->second.first].push_back(node_local_ind);
            }
            else
            {     
                treated_ranks_to_pair_ind_size_recv.insert
                (
                    std::pair<int,std::pair<int,int>>(oEdge.target_rank,{ranks_recv.size(),0})
                );
                ranks_recv.push_back(oEdge.target_rank);
                rank_ind_to_local_NodeID.push_back({node_local_ind});
            }
            treated_ranks_to_pair_ind_size_recv[oEdge.target_rank].second++;
        }
    }
    std::vector<std::vector<std::uint64_t>> rank_ind_to_area_ind_list_recv(ranks_recv.size());
    for(int i=0; i<ranks_recv.size(); i++)
    {
        int rank = ranks_recv[i];
        rank_ind_to_area_ind_list_recv[i].resize(treated_ranks_to_pair_ind_size_recv[rank].second);
    }
    
    //setup nonblocking recv for areaID data from other ranks
    std::unique_ptr<MPI_Request[]> requestArrayRecv(new MPI_Request[ranks_recv.size()]);
    for(int i=0; i<ranks_recv.size(); i++)
    {
        int source_rank = ranks_recv[i];
        int ind = treated_ranks_to_pair_ind_size_recv[source_rank].first;
        if(source_rank == my_rank)
        {
            ownRankRecvInd = ind;
            continue;
        }
        int count = treated_ranks_to_pair_ind_size_recv[source_rank].second;
        std::uint64_t* buffer = rank_ind_to_area_ind_list_recv[ind].data();
        int tag = stoi(std::to_string(source_rank)+std::to_string(my_rank));
        MPI_Request *request = requestArrayRecv.get() + i;
        MPIWrapper::Irecv(buffer,count,MPI_UINT64_T,source_rank,tag,request);
    }
    
    //prepare areaID data to send to other ranks
    std::unordered_map<int,int> treated_ranks_send;
    std::vector<std::vector<std::uint64_t>> rank_ind_to_area_ind_list_send;
    std::vector<int> ranks_send;
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        const std::vector<InEdge>& iEdges = graph.get_in_edges(my_rank,node_local_ind);
        const std::uint64_t area_localID = graph.get_node_area_localID(my_rank,node_local_ind);
        for(const InEdge& iEdge : iEdges)
        {
            const auto rank_ind = treated_ranks_send.find(iEdge.source_rank);
            if(rank_ind != treated_ranks_send.end())
            {
                rank_ind_to_area_ind_list_send[rank_ind->second].push_back(area_localID);
            }
            else
            {     
                treated_ranks_send.insert(std::pair<int,int>(iEdge.source_rank,ranks_send.size()));
                rank_ind_to_area_ind_list_send.push_back({area_localID});
                ranks_send.push_back(iEdge.source_rank);
            }
        }
    }
    
    //setup nonblocking send for areaID data to other ranks
    std::unique_ptr<MPI_Request[]> requestArraySend(new MPI_Request[ranks_recv.size()]);
    for(int i=0; i<ranks_send.size(); i++)
    {
        int target_rank = ranks_send[i];
        int ind = treated_ranks_send[target_rank];
        if(target_rank == my_rank)
        {
            rank_ind_to_area_ind_list_recv[ownRankRecvInd] = rank_ind_to_area_ind_list_send[ind];
            continue;
        }               
        int count = rank_ind_to_area_ind_list_send[ind].size();
        std::uint64_t* buffer = rank_ind_to_area_ind_list_send[ind].data();
        int tag = stoi(std::to_string(my_rank)+std::to_string(target_rank));
        MPI_Request *request = requestArraySend.get() + i;
        MPIWrapper::Isend(buffer,count,MPI_UINT64_T,target_rank,tag,request);
    }
 
    MPIWrapper::Waitall(ranks_recv.size(),requestArrayRecv.get());
    MPIWrapper::Waitall(ranks_send.size(),requestArraySend.get());

    
    AreaIDConnecMap areaIDConnecStrengthMapLocal;
    std::unordered_multimap<int,std::pair<int,int>> rankToNodeAndArea;
    
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        const AreaLocalID source_area_ID(my_rank,graph.get_node_area_localID(my_rank,node_local_ind));
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_local_ind);
        for(const OutEdge& oEdge : oEdges)
        {
            std::uint64_t otherRank = oEdge.target_rank;
            std::uint64_t otherID = oEdge.target_id;
            
            int ind = treated_ranks_to_pair_ind_size_recv[otherRank].first;
            
            const AreaLocalID target_area_ID(otherRank,-1);
            
            std::pair<AreaLocalID,AreaLocalID> area_to_area;

            
            // Non existend key has the value 0 because of value initialization
            // https://en.cppreference.com/w/cpp/container/unordered_map/operator_at
            //areaIDConnecStrengthMapLocal[area_to_area]+=oEdge.weight;
        }
    }

    std::vector<std::string> sourceArea;
    std::vector<int> sourceAreaStringLen;
    std::vector<std::string> targetArea;
    std::vector<int> targetAreaStringLen;
    std::vector<int> weightSum;
    int nbrOfEntries = sourceArea.size();
    
    /*
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
    */

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
