#include "Modularity.h"

double Modularity::computeModularity
(
    const DistributedGraph& graph
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const int number_ranks = MPIWrapper::get_number_ranks();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    const std::vector<std::string>& area_names = graph.get_local_area_names();
    
//Gather area names to all ranks
    std::function<std::unique_ptr<std::vector<std::pair<std::string,int>>>(const DistributedGraph&)> 
        get_Data = [&](const DistributedGraph& dg)
        {
            const std::vector<std::string>& area_names = dg.get_local_area_names();
            auto data = std::make_unique<std::vector<std::pair<std::string,int>>>(area_names.size());
            std::transform(area_names.cbegin(),area_names.cend(),data->begin(),
                           [](std::string name)
                            {return std::pair<std::string,int>(name,name.size());});            
            return std::move(data);
        };
    std::unique_ptr<std::vector<std::vector<std::string>>> area_names_list_of_ranks = CommunicationPatterns::gather_Data_to_all_Ranks<std::string,char>
    (
        graph,
        get_Data,
        [](std::string area_name){return std::vector<char>(area_name.cbegin(),area_name.cend());},
        [](std::vector<char>area_name_v){return std::string(area_name_v.data(),area_name_v.size());},
        MPI_CHAR
    );
        
//Create global area name indices
    std::unordered_map<std::string,std::uint64_t> area_names_map;
    std::vector<std::string> area_names_list;
    for(const std::vector<std::string>& rank_names : *area_names_list_of_ranks)
    {
        for(const std::string& name : rank_names)
        {
            auto status = area_names_map.insert(std::pair<std::string,std::uint64_t>(name,area_names_list.size()));
            if(status.second)
            {
                area_names_list.push_back(name);
            }
        }
    }
    
//Compute adjacency results
    std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>>>
                 (const DistributedGraph& dg,std::uint64_t node_localID)>
        collect_adjacency_area_info = [&](const DistributedGraph& dg,std::uint64_t node_localID)
        {
            const int my_rank = MPIWrapper::get_my_rank();
            std::uint64_t node_area_localID = dg.get_node_area_localID(my_rank, node_localID);
            auto key_Value = area_names_map.find(area_names[node_area_localID]);
            assert(key_Value!=area_names_map.end()); 
            std::uint64_t area_globalID = key_Value->second;
            
            auto outward_node_area = std::make_unique<std::vector<std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>>>();
            
            const std::vector<OutEdge>& oEdges = dg.get_out_edges(my_rank,node_localID);
            for(const OutEdge& oEdge : oEdges)
            {
                std::uint64_t rank = oEdge.target_rank;
                std::uint64_t id = oEdge.target_id;
                outward_node_area->push_back(std::tie<std::uint64_t,std::uint64_t,std::uint64_t>
                                                        (rank,id,area_globalID));
            }
            return outward_node_area;
        };
    std::function<std::uint8_t(const DistributedGraph& dg,std::uint64_t node_localID,std::uint64_t area_globalID)> 
        test_for_adjacent_equal_Area = 
                [&](const DistributedGraph& dg,std::uint64_t node_localID,std::uint64_t area_globalID)
        {
            const int my_rank = MPIWrapper::get_my_rank();            
            std::uint64_t node_area_localID = dg.get_node_area_localID(my_rank, node_localID);
            auto key_Value = area_names_map.find(area_names[node_area_localID]);
            assert(key_Value!=area_names_map.end());            
            if(area_globalID==key_Value->second)
            {
                return 1;
            }
            return 0;
        };
    std::unique_ptr<CommunicationPatterns::NodeToNodeQuestionStructure<std::uint64_t,std::uint8_t>> adjacency_results;
    adjacency_results = CommunicationPatterns::node_to_node_question<std::uint64_t,std::uint8_t>
                            (graph,MPI_UINT64_T,collect_adjacency_area_info,
                                   MPI_UINT8_T,test_for_adjacent_equal_Area);
                            
    std::uint64_t local_adjacency_sum = 0;
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        std::unique_ptr<std::vector<std::uint8_t>> this_node_adjacency_results;
        this_node_adjacency_results = adjacency_results->getAnswersOfQuestionerNode(node_local_ind);
        
        for(int i=0;i<this_node_adjacency_results->size();i++)
        {
            local_adjacency_sum += (*this_node_adjacency_results)[i];
        }
    }
    std::uint64_t global_adjacency_sum = MPIWrapper::all_reduce<std::uint64_t>(local_adjacency_sum,MPI_UINT64_T,MPI_SUM);
    
//Compute total number of edges
    std::uint64_t local_m = number_local_nodes;
    std::uint64_t global_m = MPIWrapper::all_reduce<std::uint64_t>(local_m,MPI_UINT64_T,MPI_SUM);
    if(global_m==0)
    {
        throw std::logic_error("Total number of edges must not be zero");
    }
    
//Compute in-out degree summation    
    std::vector<std::vector<NodeModularityInfo>> area_globalID_to_node(area_names_list.size());
    for(std::uint64_t node_localID=0; node_localID<number_local_nodes; node_localID++)
    {         
        std::uint64_t node_area_localID = graph.get_node_area_localID(my_rank, node_localID);
        auto key_value = area_names_map.find(area_names[node_area_localID]);
        assert(key_value!=area_names_map.end());
        std::uint64_t area_globalID = key_value->second;
    
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_localID);
        const std::vector<InEdge>& iEdges = graph.get_in_edges(my_rank,node_localID);
        NodeModularityInfo node_info;
        node_info.node_in_degree = iEdges.size();
        node_info.node_out_degree = oEdges.size();
        node_info.area_globalID = area_globalID;
        
        area_globalID_to_node[area_globalID].push_back(node_info);
    }
    std::vector<std::vector<NodeModularityInfo>> local_area_globalID_to_node;
    for(int area_globalID=0; area_globalID<area_names_list.size(); area_globalID++)
    {
        int calculation_rank = area_globalID%number_ranks;
        int area_local_size = area_globalID_to_node[area_globalID].size();
        
        std::function<std::unique_ptr<std::vector<std::pair<NodeModularityInfo,int>>>(const DistributedGraph&)> 
            get_modularity_data = [&](const DistributedGraph& dg)
            {
                auto data = std::make_unique<std::vector<std::pair<NodeModularityInfo,int>>>(area_local_size);
                std::transform(area_globalID_to_node[area_globalID].cbegin(), area_globalID_to_node[area_globalID].cend(),
                               data->begin(),[](NodeModularityInfo modulInfo)
                                    {return std::pair<NodeModularityInfo,int>(modulInfo,1);});            
                return std::move(data);
            };
        
        std::unique_ptr<std::vector<std::vector<NodeModularityInfo>>> local_area_globalID_to_nodeID = CommunicationPatterns::gather_Data_to_one_Rank<NodeModularityInfo,NodeModularityInfo>
        (
            graph,
            get_modularity_data,
            [](NodeModularityInfo modulInfo){return std::vector<NodeModularityInfo>({modulInfo});},
            [](std::vector<NodeModularityInfo> node_info_vec){return node_info_vec[0];},
            MPIWrapper::MPI_nodeModularityInfo,
            calculation_rank
        );
        if(my_rank==calculation_rank)
        {
            local_area_globalID_to_node.push_back({});
            std::for_each(local_area_globalID_to_nodeID->begin(),local_area_globalID_to_nodeID->end(),
                       [&](std::vector<NodeModularityInfo> modul_info_vec)
                       {local_area_globalID_to_node.back().insert(local_area_globalID_to_node.back().end(),
                                                                 modul_info_vec.begin(),modul_info_vec.end());});
        }
    }
    double local_in_out_degree_node_sum = 0;
    for(const std::vector<NodeModularityInfo>& nodes_of_area : local_area_globalID_to_node)
    {
        for(int i=0;i<nodes_of_area.size();i++)
        {
            for(int j=0;j<nodes_of_area.size();j++)
            {
                assert(nodes_of_area[i].area_globalID==nodes_of_area[j].area_globalID);
                local_in_out_degree_node_sum-=static_cast<double>((nodes_of_area[i].node_in_degree*nodes_of_area[j].node_out_degree))/global_m;
            }
        }
    }
    double global_in_out_degree_node_sum = MPIWrapper::all_reduce<double>(local_in_out_degree_node_sum,MPI_DOUBLE,MPI_SUM);
    
//Compute modularity of number of edges, global adjacency and global in and out degree sums
    return (static_cast<double>(global_adjacency_sum) + global_in_out_degree_node_sum)/static_cast<double>(global_m);
}

double Modularity::computeModularitySingleProc
(
    const DistributedGraph& graph
)
{
    // Define all relevant local variables
    unsigned int resultToRank = 0;
    const int& my_rank = MPIWrapper::get_my_rank();
    const int& number_ranks = MPIWrapper::get_number_ranks();
    const int& number_local_nodes = graph.get_number_local_nodes();
    std::vector<uint64_t> node_numbers = MPIWrapper::all_gather(graph.get_number_local_nodes());
    const std::vector<std::string> area_names = graph.get_local_area_names();
    const int& number_area_names = area_names.size();
    MPIWrapper::barrier();
    
    // ========== GET AREA NAMES TO SINGLE COMPUTATION RANK ==========

    // Create local transmit_area_names_string and area_names_char_len vectors
    std::vector<int> area_names_char_len;
    std::vector<char> transmit_area_names_string;
    for(const std::string& name : area_names)
    {
        for(int i = 0; i < name.size(); i++)
        {
            transmit_area_names_string.push_back(name[i]);
        }
        area_names_char_len.push_back(name.size());
    }
    MPIWrapper::barrier();
    
    // Gather rank_to_number_area_names vector as helper for char_len_displ (and global_area_names_char_len)
    assert(area_names_char_len.size() == number_area_names); //debug
    int nbr_area_names = area_names.size();
    std::vector<int> rank_to_number_area_names;
    if(my_rank == resultToRank)
    {
        rank_to_number_area_names.resize(number_ranks);
    }
    MPIWrapper::gather<int>(&nbr_area_names, rank_to_number_area_names.data(), 1, MPI_INT, resultToRank);
    MPIWrapper::barrier();

    // Crate char_len_displ as helper for global_area_names_char_len 
    // and prepare global_area_names_char_len with correct size
    std::vector<int> char_len_displ;
    std::vector<int> global_area_names_char_len;
    if(my_rank == resultToRank)
    {
        char_len_displ.resize(number_ranks);
        int displacement = 0;
        
        for(int r = 0; r < number_ranks; r++)
        {
            char_len_displ[r] = displacement;
            displacement += rank_to_number_area_names[r];
        }
        global_area_names_char_len.resize(displacement);
    }
    MPIWrapper::barrier();

    // Gather global_area_names_char_len (with Help of char_len_displ and rank_to_number_area_names)
    MPIWrapper::gatherv<int>(area_names_char_len.data(), nbr_area_names,
                            global_area_names_char_len.data(), rank_to_number_area_names.data(), char_len_displ.data(),
                            MPI_INT, resultToRank);
    MPIWrapper::barrier();

    // Finally create rank_to_area_names_char_len as helper for rank_to_area_names
    std::vector<std::vector<int>> rank_to_area_names_char_len;
    if(my_rank==resultToRank)
    {
        rank_to_area_names_char_len.resize(number_ranks);
        for(int r = 0; r < number_ranks-1; r++)
        {
            for(int l = char_len_displ[r]; l < char_len_displ[r+1]; l++)
            {
                rank_to_area_names_char_len[r].push_back(global_area_names_char_len[l]);
            }
        }
        for(int l = char_len_displ[number_ranks-1]; l < global_area_names_char_len.size(); l++)
        {
            rank_to_area_names_char_len[number_ranks-1].push_back(global_area_names_char_len[l]);
        }
    }
    MPIWrapper::barrier();

    // Gather rank_to_string_len as helper for char_displ
    int nbr_string_chars = transmit_area_names_string.size();
    std::vector<int> rank_to_string_len;
    if(my_rank == resultToRank)
    {
        rank_to_string_len.resize(number_ranks);
    }
    MPIWrapper::gather<int>(&nbr_string_chars, rank_to_string_len.data(), 1, MPI_INT, resultToRank);
    MPIWrapper::barrier();

    // Create char_displ as helper for rank_to_area_names
    std::vector<int> char_displ;
    if(my_rank == resultToRank)
    {
        char_displ.resize(number_ranks);
        int displacement = 0;
        for(int r = 0; r < number_ranks; r++)
        {
                char_displ[r] = displacement;
                displacement += rank_to_string_len[r];
        }
    }
    MPIWrapper::barrier();

    // Prepare and gather global_area_names_string as helper for rank_to_area_names
    std::vector<char> global_area_names_string;
    if(my_rank == resultToRank)
    {
        int sum = std::accumulate(rank_to_string_len.begin(),
                                    rank_to_string_len.end(), 0);
        global_area_names_string.resize(sum);
    }
    MPIWrapper::gatherv<char>(transmit_area_names_string.data(), nbr_string_chars,
                              global_area_names_string.data(), rank_to_string_len.data(), char_displ.data(),
                              MPI_CHAR, resultToRank);
    MPIWrapper::barrier();
    
    // Finally create rank_to_area_names
    std::vector<std::vector<std::string>> rank_to_area_names;
    if(my_rank == resultToRank)
    {
        rank_to_area_names.resize(number_ranks);
        int displacement = 0;
        for(int r = 0; r < rank_to_area_names_char_len.size(); r++)
        {
            for(int l = 0; l < rank_to_area_names_char_len[r].size(); l++)
            {
                std::string name(&global_area_names_string[displacement],
                                 rank_to_area_names_char_len[r][l]);
                rank_to_area_names[r].push_back(name);
                displacement += rank_to_area_names_char_len[r][l];
            }
        }
    }
    MPIWrapper::barrier();

    // ========== LET SINGLE RANK COMPUTE MODULARITY ==========
    double result;

    std::uint64_t local_m = number_local_nodes;
    std::uint64_t m = MPIWrapper::all_reduce<std::uint64_t>(local_m, MPI_UINT64_T, MPI_SUM);
    std::cout<<"singleProc global m:"<<m<<std::endl;

    // Computation is performed by a single process:
    if(my_rank == resultToRank)
    {
        double sum = 0.0;
        
        for(int ir = 0; ir < number_ranks; ir++)
        {
            for(int i = 0; i < node_numbers[ir]; i++)
            {
                std::uint64_t i_area_localID = graph.get_node_area_localID(ir, i);
                std::string i_area_str = rank_to_area_names[ir][i_area_localID];
                int ki_in = graph.get_in_edges(ir, i).size();
                const std::vector<OutEdge>& i_out_edges = graph.get_out_edges(ir, i);
                for(int jr = 0; jr < number_ranks; jr++)
                {
                    for(int j = 0; j < node_numbers[jr]; j++)
                    {
                        
                        std::uint64_t j_area_localID = graph.get_node_area_localID(jr, j);
                        std::string j_area_str = rank_to_area_names[jr][j_area_localID];
                        if(i_area_str == j_area_str)
                        {
                            int kj_out = graph.get_out_edges(jr, j).size();
                            int a_ij = 0; 
                            for(const OutEdge& i_out_edge : i_out_edges)
                            { 
                                if(i_out_edge.target_rank == jr && i_out_edge.target_id == j){
                                    a_ij = 1;
                                }
                            }
                            sum += a_ij - static_cast<double>((ki_in * kj_out))/m;
                        }
                    }
                }
            }
            std::cout << "i_rank(" << ir << "/" << number_ranks << ") with " << node_numbers[ir] << " nodes" << std::endl; 
        }
        result = sum/m;

        // Print out AreaConnecMap:
        std::cout << "Modularity (serial): " << result << std::endl;
    }
    return result;
}
