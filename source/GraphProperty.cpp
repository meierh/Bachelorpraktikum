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
    std::vector<std::unordered_map<std::uint64_t,int>> rank_ind_NodeID_to_localInd;
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_local_ind);
        for(const OutEdge& oEdge : oEdges)
        {
            const auto rank_ind = treated_ranks_to_pair_ind_size_recv.find(oEdge.target_rank);
            if(rank_ind != treated_ranks_to_pair_ind_size_recv.end())
            {
                rank_ind_NodeID_to_localInd[rank_ind->second.first].insert
                (
                    std::pair<std::uint64_t,int>
                    (
                        node_local_ind,
                        treated_ranks_to_pair_ind_size_recv[oEdge.target_rank].second
                    )
                );
            }
            else
            {
                treated_ranks_to_pair_ind_size_recv.insert
                (
                    std::pair<int,std::pair<int,int>>(oEdge.target_rank,{ranks_recv.size(),0})
                );
                ranks_recv.push_back(oEdge.target_rank);
                rank_ind_NodeID_to_localInd.push_back({{node_local_ind,0}});
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
 
    //Wait for send and recv completion
    MPIWrapper::Waitall(ranks_recv.size(),requestArrayRecv.get());
    MPIWrapper::Waitall(ranks_send.size(),requestArraySend.get());
    
    //Create rank local area distance sum
    AreaIDConnecMap areaIDConnecStrengthMapLocal;    
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        const AreaLocalID source_area_ID(my_rank,graph.get_node_area_localID(my_rank,node_local_ind));
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_local_ind);
        for(const OutEdge& oEdge : oEdges)
        {
            std::uint64_t otherRank = oEdge.target_rank;
            std::uint64_t otherID = oEdge.target_id;
            
            int rank_local_ind = treated_ranks_to_pair_ind_size_recv[otherRank].first;
            int nodeID_localInd = rank_ind_NodeID_to_localInd[rank_local_ind][node_local_ind];
            
            AreaLocalID target_area_ID(otherRank,rank_ind_to_area_ind_list_recv[rank_local_ind][nodeID_localInd]);
            std::pair<AreaLocalID,AreaLocalID> area_to_area(source_area_ID,target_area_ID);

            // Non existend key has the value 0 because of value initialization
            // https://en.cppreference.com/w/cpp/container/unordered_map/operator_at
            areaIDConnecStrengthMapLocal[area_to_area]+=oEdge.weight;
        }
    }
    
    //Exchange of area local ind and strings
    std::unordered_map<AreaLocalID,std::string,stdPair_hash> areaID_to_name;
    std::vector<int> area_names_char_dist;
    const std::vector<std::string> area_names = graph.get_area_names();
    for(const std::string& name : area_names)
    {
        area_names_char_dist.push_back(name.size());
    }
    int area_names_totalLength = std::accumulate(area_names_char_dist.begin(),area_names_char_dist.end(),0);
    auto transmit_area_names = std::make_unique<char[]>(area_names_totalLength);
    if(my_rank==resultToRank)
    {
        for(int otherRank=0; otherRank<MPIWrapper::get_number_ranks(); otherRank++)
        {
            if(otherRank==resultToRank)
                continue;
            
            int nbr_of_area_names;
            MPIWrapper::Recv(&nbr_of_area_names,1,MPI_INT,otherRank,0);
            std::vector<int> area_names_char_dist_other(nbr_of_area_names);
            MPIWrapper::Recv(area_names_char_dist_other.data(),nbr_of_area_names,MPI_INT,otherRank,1);
            int area_names_totalLength_other = std::accumulate(area_names_char_dist.begin(),area_names_char_dist.end(),0);
            auto transmit_area_names_other = std::make_unique<char[]>(area_names_totalLength_other);
            MPIWrapper::Recv(transmit_area_names_other.get(),area_names_totalLength_other,MPI_CHAR,otherRank,3);
            std::vector<std::string> area_names_other(nbr_of_area_names);
            int pChar=0;
            for(int i=0;i<area_names_other.size();i++)
            {
                area_names_other[i] = std::string(transmit_area_names_other.get()+pChar,area_names_char_dist_other[i]);
                pChar+=area_names_char_dist_other[i];
            }
            for(int i=0;i<area_names_other.size();i++)
            {
                areaID_to_name.insert({{otherRank,i},area_names_other[i]});
            }
        }
        for(int i=0;i<area_names.size();i++)
        {
            areaID_to_name.insert({{my_rank,i},area_names[i]});
        }
    }
    else
    {
        int size = area_names_char_dist.size();
        MPIWrapper::Send(&size,1,MPI_INT,resultToRank,0);
        MPIWrapper::Send(area_names_char_dist.data(),area_names_char_dist.size(),MPI_INT,resultToRank,1);
        MPIWrapper::Send(transmit_area_names.get(),area_names_totalLength,MPI_CHAR,resultToRank,3);
    }
    
    //Send areaID Connec Sums    
    if(my_rank==resultToRank)
    {
        for(int otherRank=0; otherRank<MPIWrapper::get_number_ranks(); otherRank++)
        {
            if(otherRank==resultToRank)
                continue;
            
            int nbr_of_area_names;
            MPIWrapper::Recv(&nbr_of_area_names,1,MPI_INT,otherRank,4);
            std::vector<std::pair<AreaLocalID,AreaLocalID>> area_to_area_list(nbr_of_area_names);
            std::vector<int> weightSum_list(nbr_of_area_names);
            int nbrBytes = nbr_of_area_names*sizeof(std::pair<AreaLocalID,AreaLocalID>);
            MPIWrapper::Recv(area_to_area_list.data(),nbrBytes,MPI_BYTE,otherRank,5);
            MPIWrapper::Recv(weightSum_list.data(),nbr_of_area_names,MPI_INT,otherRank,6);
            
            for(int i=0;i<nbr_of_area_names;i++)
            {
                areaIDConnecStrengthMapLocal[area_to_area_list[i]]+=weightSum_list[i];
            }
        }        
    }
    else
    {
        std::vector<std::pair<AreaLocalID,AreaLocalID>> area_to_area_list;
        std::vector<int> weightSum_list;
        for(auto keyValue = areaIDConnecStrengthMapLocal.begin();
            keyValue != areaIDConnecStrengthMapLocal.end();
            ++keyValue)
        {
            area_to_area_list.push_back(keyValue->first);
            weightSum_list.push_back(keyValue->second);
        }
        int size = area_to_area_list.size();
        MPIWrapper::Send(&size,1,MPI_INT,resultToRank,4);
        int nbrBytes = size*sizeof(std::pair<AreaLocalID,AreaLocalID>);
        MPIWrapper::Send(area_to_area_list.data(),nbrBytes,MPI_BYTE,resultToRank,5);
        MPIWrapper::Send(weightSum_list.data(),size,MPI_INT,resultToRank,6);
    }
    
    //Transfer data from ID to string
    auto result = std::make_unique<AreaConnecMap>();;
    if(my_rank==resultToRank)
    {
        for(auto keyValue = areaIDConnecStrengthMapLocal.begin();
            keyValue != areaIDConnecStrengthMapLocal.end();
            ++keyValue)
        {
            AreaLocalID source_area_ID = keyValue->first.first;
            AreaLocalID target_area_ID = keyValue->first.second;
            std::string source_area_name = areaID_to_name[source_area_ID];
            std::string target_area_name = areaID_to_name[target_area_ID];
            (*result)[{source_area_name,target_area_name}]+=keyValue->second;
        }        
    }
    
    return std::move(result);
}
