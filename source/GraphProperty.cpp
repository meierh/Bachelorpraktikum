#include "GraphProperty.h"

std::unique_ptr<GraphProperty::AreaConnecMap> GraphProperty::areaConnectivityStrength
(
    const DistributedGraph& graph,
    const unsigned int resultToRank
)
{
// Test function parameters
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    if(resultToRank>=number_of_ranks)
    {
        throw std::invalid_argument("Bad parameter - resultToRank:"+resultToRank);
    }
    
// Build local area connection map
    const int my_rank = MPIWrapper::get_my_rank();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,areaConnectivityInfo>>>
        (const DistributedGraph&, std::uint64_t)>
        transfer_connection_sources = [&](const DistributedGraph& dg,std::uint64_t node_local_ind)
        {
            const std::vector<OutEdge>& oEdges = dg.get_out_edges(my_rank,node_local_ind);
            std::int64_t node_areaID = dg.get_node_area_localID(my_rank,node_local_ind);
            auto connection_sources=
            std::make_unique<std::vector<std::tuple<std::uint64_t,std::uint64_t,areaConnectivityInfo>>>(oEdges.size());
            for(int i=0; i<oEdges.size(); i++)
            {
                areaConnectivityInfo one_connection = {my_rank,node_areaID,oEdges[i].target_rank,-1,oEdges[i].weight};
                (*connection_sources)[i] = std::tie(oEdges[i].target_rank,oEdges[i].target_id,one_connection);
            }
            return std::move(connection_sources);
        };
    std::function<areaConnectivityInfo(const DistributedGraph&, std::uint64_t, areaConnectivityInfo)> 
        fill_connection_target_areaID = 
        [&](const DistributedGraph& dg,std::uint64_t node_local_ind,areaConnectivityInfo connection)
        {
            connection.target_area_localID = dg.get_node_area_localID(my_rank,node_local_ind);
            return connection;
        };
    std::unique_ptr<NodeToNodeQuestionStructure<areaConnectivityInfo,areaConnectivityInfo>> area_connections=
        node_to_node_question<areaConnectivityInfo,areaConnectivityInfo>
                (graph,MPIWrapper::MPI_areaConnectivityInfo,transfer_connection_sources,
                 MPIWrapper::MPI_areaConnectivityInfo,fill_connection_target_areaID);
    AreaIDConnecMap my_rank_connecID_map;
    for(std::uint64_t node_ID = 0;node_ID<number_local_nodes;node_ID++)
    {
        std::unique_ptr<std::vector<areaConnectivityInfo>> node_connections = 
            area_connections->getAnswersOfQuestionerNode(node_ID);
        for(areaConnectivityInfo& one_connection : *node_connections)
        {
            AreaLocalID source_nameID(one_connection.source_rank,one_connection.source_area_localID);
            AreaLocalID target_nameID(one_connection.target_rank,one_connection.target_area_localID);
            my_rank_connecID_map[{source_nameID,target_nameID}] += one_connection.weight;
        }
    }

// Gather local area connection maps to result rank
    using connecID_map_data = std::tuple<std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t>;
    std::function<std::unique_ptr<std::vector<std::pair<connecID_map_data,int>>>(const DistributedGraph&)> 
        extract_connecID_map = 
        [&](const DistributedGraph& dg)
        {
            auto connecID_list = std::make_unique<std::vector<std::pair<connecID_map_data,int>>>();
            for(auto map_entry=my_rank_connecID_map.cbegin();map_entry!=my_rank_connecID_map.cend();map_entry++)
            {
                auto connecID_data_Entry = std::tie(map_entry->first.first.first ,map_entry->first.first.second,
                                                    map_entry->first.second.first,map_entry->first.second.second,
                                                    map_entry->second);
                std::pair<connecID_map_data,int> composed_Entry(connecID_data_Entry,std::tuple_size<connecID_map_data>());
                connecID_list->push_back(composed_Entry);
            }
            return connecID_list;
        };    
    std::unique_ptr<std::vector<std::vector<connecID_map_data>>> ranks_to_connecID_data = 
    gather_Data_to_one_Rank<connecID_map_data,std::int64_t>
    (
        graph,
        extract_connecID_map,
        [](connecID_map_data dat){auto [s_r,s_n,t_r,t_n,w] = dat; return std::vector<std::int64_t>({s_r,s_n,t_r,t_n,w});},
        [](std::vector<std::int64_t>& data_vec)
            {
                assert(data_vec.size()==std::tuple_size<connecID_map_data>());
                return std::tie(data_vec[0],data_vec[1],data_vec[2],data_vec[3],data_vec[4]);
            },
        MPI_INT64_T,
        resultToRank
    );
    
// Gather name lists from all ranks to result rank
    std::function<std::unique_ptr<std::vector<std::pair<std::string,int>>>(const DistributedGraph&)> 
        get_names = [](const DistributedGraph& dg)
        {
            const std::vector<std::string>& area_names = dg.get_local_area_names();
            auto data = std::make_unique<std::vector<std::pair<std::string,int>>>(area_names.size());
            std::transform(area_names.cbegin(),area_names.cend(),data->begin(),[](std::string name){return std::pair<std::string,int>(name,name.size());});            
            return std::move(data);
        };
    std::unique_ptr<std::vector<std::vector<std::string>>> area_names_list_of_ranks = gather_Data_to_one_Rank<std::string,char>
    (
        graph,
        get_names,
        [](std::string area_name){return std::vector<char>(area_name.cbegin(),area_name.cend());},
        [](std::vector<char>area_name_v){return std::string(area_name_v.data(),area_name_v.size());},
        MPI_CHAR,
        resultToRank
    );
    
// Combine connecID_maps and transfer IDs to area names
    auto global_connecName_map = std::make_unique<AreaConnecMap>();
    for(const std::vector<connecID_map_data>& connecData_of_rank : *ranks_to_connecID_data)
    {
        for(const connecID_map_data& connec_data : connecData_of_rank)
        {
            assert(std::get<0>(connec_data)<area_names_list_of_ranks->size());
            assert(std::get<1>(connec_data)<(*area_names_list_of_ranks)[std::get<0>(connec_data)].size());
            std::string source_area = (*area_names_list_of_ranks)[std::get<0>(connec_data)][std::get<1>(connec_data)];
            assert(std::get<2>(connec_data)<area_names_list_of_ranks->size());
            assert(std::get<3>(connec_data)<(*area_names_list_of_ranks)[std::get<2>(connec_data)].size());
            std::string target_area = (*area_names_list_of_ranks)[std::get<2>(connec_data)][std::get<3>(connec_data)];
            (*global_connecName_map)[{source_area,target_area}] += std::get<4>(connec_data);
        }
    }

    return std::move(global_connecName_map);
}

std::unique_ptr<GraphProperty::AreaConnecMap> GraphProperty::areaConnectivityStrengthSingleProc_Helge
(
    const DistributedGraph& graph,
    unsigned int resultToRank
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    
    std::function<std::unique_ptr<std::vector<std::pair<std::uint64_t,int>>>(const DistributedGraph& dg)> 
        get_number_nodes = 
        [&](const DistributedGraph& dg)
        {
            auto number_nodes = std::make_unique<std::vector<std::pair<std::uint64_t,int>>>();
            number_nodes->push_back(std::pair<std::uint64_t,int>(number_local_nodes,1));
            return number_nodes;
        };    
    std::unique_ptr<std::vector<std::vector<std::uint64_t>>> ranks_to_number_local_nodes = 
    gather_Data_to_one_Rank<std::uint64_t,std::uint64_t>
    (
        graph,
        get_number_nodes,
        [](std::uint64_t dat){return std::vector<std::uint64_t>({dat});},
        [](std::vector<std::uint64_t>& data_vec){return data_vec[0];},
        MPI_UINT64_T,
        resultToRank    
    );
    
    MPIWrapper::barrier();
    std::cout<<"Hello from 174 "<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();  
    
    std::function<std::unique_ptr<std::vector<std::pair<std::string,int>>>(const DistributedGraph&)> 
        getNames = [](const DistributedGraph& dg)
        {
            const std::vector<std::string>& area_names = dg.get_local_area_names();
            auto data = std::make_unique<std::vector<std::pair<std::string,int>>>(area_names.size());
            std::transform(area_names.cbegin(),area_names.cend(),data->begin(),
                           [](std::string name){return std::pair<std::string,int>(name,name.size());}
                           );
            return std::move(data);
        };
    std::unique_ptr<std::vector<std::vector<std::string>>> area_names_list_of_ranks = gather_Data_to_one_Rank<std::string,char>
    (
        graph,
        getNames,
        [](std::string area_name){return std::vector<char>(area_name.cbegin(),area_name.cend());},
        [](std::vector<char>area_name_v){return std::string(area_name_v.data(),area_name_v.size());},
        MPI_CHAR,
        resultToRank
    );
    
    MPIWrapper::barrier();
    std::cout<<"Hello from 199 "<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();  
    
    auto global_connecName_map = std::make_unique<AreaConnecMap>();
    if(my_rank==resultToRank)
    {
        for(int rank=0;rank<number_of_ranks;rank++)
        {
            for(std::uint64_t node_local_ind=0; node_local_ind<(*ranks_to_number_local_nodes)[rank][0];node_local_ind++)
            {
                std::int64_t source_node_areaID = graph.get_node_area_localID(rank,node_local_ind);
                const std::vector<OutEdge>& oEdges = graph.get_out_edges(rank,node_local_ind);
                for(const OutEdge& oEdge: oEdges)
                {
                    std::int64_t target_node_areaID = graph.get_node_area_localID(oEdge.target_rank,oEdge.target_id);
                    std::string source_area = (*area_names_list_of_ranks)[rank][source_node_areaID];
                    std::string target_area = (*area_names_list_of_ranks)[oEdge.target_rank][target_node_areaID];
                    (*global_connecName_map)[{source_area,target_area}] += oEdge.weight;
                }
            }
        }
    }
    MPIWrapper::barrier();
    std::cout<<"Hello from 223 "<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();  
    return std::move(global_connecName_map);
}

std::unique_ptr<GraphProperty::AreaConnecMap> GraphProperty::areaConnectivityStrengthSingleProc
(
    const DistributedGraph& graph,
    unsigned int resultToRank
)
{
    // Define all relevant local variables
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
        
        // Print out rank_to_area_names
        for(int i = 0; i < rank_to_area_names.size(); i++)
        {
            std::cout<<"From Rank: "<<i<<" [";
            for(int j=0;j<rank_to_area_names[i].size();j++)
            {
                std::cout<<" "<<rank_to_area_names[i][j];
            }
            std::cout<<"] "<<rank_to_area_names[i].size()<<std::endl;
        }
        std::cout << "" << std::endl;
    }
    MPIWrapper::barrier();
    
    // ========== LET SINGLE RANK COMPUTE AREA CONNECTIVITY ==========

    auto result = std::make_unique<AreaConnecMap>();

    // Computation is performed by a single process:
    if(my_rank == resultToRank)
    {
        // Iterate over each rank...
        for(int r = 0; r < number_ranks; r++)
        {
            // ...and over each node from that rank
            for(int n = 0; n < node_numbers[r]; n++)
            {
                // Consider each outgoing edge and find out source and target areas
                const std::vector<OutEdge>& oEdges = graph.get_out_edges(r, n);
                for(const OutEdge& oEdge : oEdges)
                {
                    std::uint64_t source_area_localID = graph.get_node_area_localID(r, n);
                    std::uint64_t target_area_localID = graph.get_node_area_localID(oEdge.target_rank, oEdge.target_id);
                    std::string source_area_str = rank_to_area_names[r][source_area_localID];
                    std::string target_area_str = rank_to_area_names[oEdge.target_rank][target_area_localID];
                    
                    // Store all weights of the area pairs that realize a connection of two different areas 
                    // in the corresponding "area to area hash class" of the result map
                    (*result)[{source_area_str, target_area_str}] += oEdge.weight;
                }
            }
        }
        // Print out AreaConnecMap:
        int nr = 0;
        for (auto& [key, value]: (*result)) 
        {
            std::cout << "connection " << nr << 
                ": weight = " << value << " (" << key.first << " --> " << key.second << ")" << std::endl;
            nr++;
        }
        std::cout << "Elements in areaConnecMap (single proc): " << nr << std::endl;
        std::cout << "Size of areaConnecMap (single proc): " << result->size() << std::endl;
    }
    return std::move(result);
}

bool GraphProperty::compare_area_connecs(std::unique_ptr<GraphProperty::AreaConnecMap> const &map1, std::unique_ptr<GraphProperty::AreaConnecMap> const &map2, unsigned int resultToRank)
{
    if(MPIWrapper::get_my_rank() == resultToRank)
    {    
        std::cout << "compare_area_connecs: size of map1 = " << map1->size() << std::endl;
        std::cout << "compare_area_connecs: size of map2 = " << map2->size() << std::endl;

        for (auto it1 = map1->begin(); it1 != map1->end(); it1++) 
        {
            auto it2 = map2->find(it1->first);
            if(it2->second != it1->second)
            {
                std::cout << "compare_area_connecs: map1 has elements map2 is missing" << std::endl;
                return false;
            }
        }

        for (auto it2 = map2->begin(); it2 != map2->end(); it2++) 
        {
            auto it1 = map1->find(it2->first);
            if(it1->second != it2->second)
            {
                std::cout << "compare_area_connecs: map2 has elements map1 is missing" << std::endl;
                return false;
            }
        }
        std::cout << "compare_area_connecs: map1 is equal to map2" << std::endl;
    }
    return true;
}

bool GraphProperty::compare_area_connecs_alt(std::unique_ptr<GraphProperty::AreaConnecMap> const &map1, std::unique_ptr<GraphProperty::AreaConnecMap> const &map2, unsigned int resultToRank)
{
    if(MPIWrapper::get_my_rank() == resultToRank)
    {    
        
        if(map1->size() != map2->size())
        {
            std::cout << "compare_area_connecs: map1.size=" << map1->size() << ", map2.size=" << map2->size() << std::endl;
            std::cout << "compare_area_connecs: maps have different sizes" << std::endl;
            return false;
        }
        
        int nr = 0;
        auto iter2 = map2->begin();
        for (auto iter1 = map1->begin(); iter1 != map1->end(); iter1++) 
        {   
            if(*iter1 != *iter2)    //possible debug pending if operator!= is not recursive for pairs (then commpare tuple elementwise)
            {
                std::cout << "compare_area_connecs: maps are not equal" << std::endl;
                return false;
            }
            iter2++; nr++;
        }
        std::cout << "compare_area_connecs: maps are equal" << std::endl;
    }
    return true;
}

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramm_constBinWidth
(
    const DistributedGraph& graph,
    const double bin_width,
    const unsigned int resultToRank
)
{
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    if(resultToRank>=number_of_ranks)
    {
        throw std::invalid_argument("Bad parameter - resultToRank"+resultToRank);
    }
    if(bin_width<=0)
    {
        throw std::invalid_argument("Bad parameter - bin_width");
    }
        
    std::function<std::unique_ptr<Histogram>(const double,const double)> bin_width_histogram_creator =
    [=](const double min_length, const double max_length)
    {
        double span_length = max_length-min_length;
        if(span_length<=0)
        {
            throw std::invalid_argument("Span of edge distribution must be larger than zero!");
        }
        unsigned int number_bins = std::ceil(span_length/bin_width);
        if(number_bins<1)
        {
            throw std::invalid_argument("Number of bins must be greater than zero");
        }
        double two_side_overlap_mult = span_length/bin_width - std::floor(span_length/bin_width);
        double one_side_overlap = (two_side_overlap_mult/2)*bin_width;
        double start_length = min_length-one_side_overlap;

        auto histogram = std::make_unique<Histogram>(number_bins);
        for(int i=0;i<number_bins;i++)
        {
            (*histogram)[i].first.first = start_length;
            (*histogram)[i].first.second = start_length+bin_width;
            (*histogram)[i].second = 0;
            start_length = start_length+bin_width;
        }
        return std::move(histogram);
    };
    
    return std::move(edgeLengthHistogramm(graph,bin_width_histogram_creator,resultToRank));
}

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramm_constBinCount
(
    const DistributedGraph& graph,
    const std::uint64_t bin_count,
    const unsigned int resultToRank
)
{
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    if(resultToRank>=number_of_ranks)
    {
        throw std::invalid_argument("Bad parameter - resultToRank"+resultToRank);
    }
    if(bin_count<1)
    {
        throw std::invalid_argument("Bad parameter - bin_count"+bin_count);
    }
    
    std::function<std::unique_ptr<Histogram>(const double,const double)> bin_count_histogram_creator =
    [=](const double min_length, const double max_length)
    {
        double span_length = max_length-min_length;
        if(span_length<=0)
        {
            throw std::invalid_argument("Span of edge distribution must be larger than zero!");
        }
        double bin_width = span_length/bin_count;
        double start_length = min_length;

        auto histogram = std::make_unique<Histogram>(bin_count);
        for(int i=0;i<bin_count;i++)
        {
            (*histogram)[i].first.first = start_length;
            (*histogram)[i].first.second = start_length+bin_width;
            (*histogram)[i].second = 0;
            start_length = start_length+bin_width;
        }
        //Small overlap to avoid comparison errors
        histogram->front().first.first = std::nextafter(min_length,min_length-1);
        histogram->back().first.second = std::nextafter(max_length,max_length+1);
        
        return std::move(histogram);
    };
    
    return std::move(edgeLengthHistogramm(graph,bin_count_histogram_creator,resultToRank));
}

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramm_constBinWidthSingleProc
(
    const DistributedGraph& graph,
    double bin_width,
    unsigned int resultToRank
)
{
    std::function<std::unique_ptr<Histogram>(double,double)> bin_width_histogram_creator =
    [=](double min_length, double max_length)
    {
        double span_length = max_length-min_length;
        unsigned int number_bins = std::ceil(span_length/bin_width);
        if(number_bins<1)
            throw std::invalid_argument("Number of bins must be greater than zero");
        double two_side_overlap_mult = span_length/bin_width - std::floor(span_length/bin_width);
        double one_side_overlap = (two_side_overlap_mult/2)*bin_width;
        double start_length = min_length-one_side_overlap;

        auto histogram = std::make_unique<Histogram>(number_bins);
        for(int i=0;i<number_bins;i++)
        {
            (*histogram)[i].first.first = start_length;
            (*histogram)[i].first.second = start_length+bin_width;
            (*histogram)[i].second = 0;
            start_length = start_length+bin_width;
        }
        /*
        // Print out histogram dimensions:
        std::cout << "number of bins: " << number_bins << std::endl;
        std::cout << "bin width: " << bin_width << " (span length: " << span_length << ")" << std::endl;
        std::cout << "min edge length: " << min_length << " (bin start: " << (*histogram)[0].first.first << ")" << std::endl;
        std::cout << "max edge length: " << max_length << " (bin end: " << (*histogram)[number_bins-1].first.second << ")" << std::endl;
        */
        return std::move(histogram);
    };
    return std::move(edgeLengthHistogramSingleProc(graph,bin_width_histogram_creator,resultToRank));
}

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramm_constBinCountSingleProc
(
    const DistributedGraph& graph,
    std::uint64_t bin_count,
    unsigned int resultToRank
)
{
    if(bin_count<1)
        throw std::invalid_argument("Number of bins must be greater than zero");
    
    std::function<std::unique_ptr<Histogram>(double,double)> bin_count_histogram_creator =
    [=](double min_length, double max_length)
    {
        double span_length = max_length-min_length;
        if(span_length<=0)
        {
            throw std::invalid_argument("Span of edge distribution must be larger than zero!");
        }
        double bin_width = span_length/bin_count;
        double start_length = min_length;

        auto histogram = std::make_unique<Histogram>(bin_count);
        for(int i=0;i<bin_count;i++)
        {
            (*histogram)[i].first.first = start_length;
            (*histogram)[i].first.second = start_length+bin_width;
            (*histogram)[i].second = 0;
            start_length = start_length+bin_width;
        }
        //Small overlap to avoid comparison errors
        histogram->front().first.first = std::nextafter(min_length,min_length-1);
        histogram->back().first.second = std::nextafter(max_length,max_length+1);
        
        return std::move(histogram);
    };
    return std::move(edgeLengthHistogramSingleProc(graph,bin_count_histogram_creator,resultToRank));
}

std::vector<long double> GraphProperty::networkTripleMotifs
(
    const DistributedGraph& graph,
    unsigned int resultToRank
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
        
    //Create rank local area distance sum
    std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,threeMotifStructure>>>
                 (const DistributedGraph& dg,std::uint64_t node_local_ind)>
        collect_possible_networkMotifs_oneNode = [](const DistributedGraph& dg,std::uint64_t node_local_ind)
        {
            const int my_rank = MPIWrapper::get_my_rank();
            
            auto this_node_possible_motifs = std::make_unique<std::vector<std::tuple<std::uint64_t,std::uint64_t,threeMotifStructure>>>();
            
            const std::vector<OutEdge>& oEdges = dg.get_out_edges(my_rank,node_local_ind);
            const std::vector<InEdge>& iEdges = dg.get_in_edges(my_rank,node_local_ind);
            
            std::unordered_map<std::pair<std::uint64_t,std::uint64_t>,
                            std::pair<bool,bool>,
                            stdPair_hash> adjacent_nodes;
            
            for(const OutEdge& oEdge : oEdges)
            {
                std::pair<std::uint64_t,std::uint64_t> node_key(oEdge.target_rank,oEdge.target_id);
                std::pair<bool,bool>& value = adjacent_nodes[node_key];
                assert(!value.first);
                value.first = true;
            }
            for(const InEdge& iEdge : iEdges)
            {
                std::pair<std::uint64_t,std::uint64_t> node_key(iEdge.source_rank,iEdge.source_id);
                std::pair<bool,bool>& value = adjacent_nodes[node_key];
                assert(!value.second);
                value.second = true;
            }
            for(auto iterOuter = adjacent_nodes.begin(); iterOuter!=adjacent_nodes.end(); iterOuter++)
            {
                std::pair<std::uint64_t,std::uint64_t> node_Outer_key = iterOuter->first;
                
                for(auto iterInner = adjacent_nodes.begin(); iterInner!=adjacent_nodes.end(); iterInner++)
                {
                    std::pair<std::uint64_t,std::uint64_t> node_Inner_key = iterInner->first;
                    
                    if(node_Inner_key != node_Outer_key)
                    {
                        //std::tuple<std::uint64_t,std::uint64_t,threeMotifStructure> possible_motif;
                        threeMotifStructure motifStruc;
                        motifStruc.node_1_rank = my_rank;
                        motifStruc.node_1_local = node_local_ind;
                        motifStruc.node_2_rank = node_Outer_key.first;
                        motifStruc.node_2_local = node_Outer_key.second;
                        motifStruc.node_3_rank = node_Inner_key.first;
                        motifStruc.node_3_local = node_Inner_key.second;
                        
                        bool node_2_exists_outEdge = iterOuter->second.first;
                        bool node_2_exists_inEdge  = iterOuter->second.second;
                        bool node_3_exists_outEdge = iterInner->second.first;
                        bool node_3_exists_inEdge  = iterInner->second.second;

                        std::uint8_t exists_edge_bitArray = 0;
                        exists_edge_bitArray |= node_2_exists_outEdge?1:0;
                        exists_edge_bitArray |= node_2_exists_inEdge?2:0;
                        exists_edge_bitArray |= node_3_exists_outEdge?4:0;
                        exists_edge_bitArray |= node_3_exists_inEdge?8:0;

                        switch (exists_edge_bitArray)
                        {
                            case 10:
                                //three node motif 1 & 11 (1010)
                                motifStruc.setMotifTypes({1,11});
                                break;
                            case 9:
                                //three node motif 2 & 7 (1001)
                                motifStruc.setMotifTypes({2,7});
                                break;
                            case 5:
                                //three node motif 3 & 5 & 8 (0101)
                                motifStruc.setMotifTypes({3,5,8});
                                break;
                            case 11:
                                //three node motif 4 (1011)
                                motifStruc.setMotifTypes({4});
                                break;
                            case 7:
                                //three node motif 6 (0111)
                                motifStruc.setMotifTypes({6});
                                break;
                            case 15:
                                //three node motif 9 & 12 & 13 (1111)
                                motifStruc.setMotifTypes({9,12,13});
                                break;
                            case 6:
                                //three node motif 10 & 7 (0110)
                                motifStruc.setMotifTypes({10,7});
                                break;
                            default:
                                assert(false);
                        }
                        
                        auto possible_motif = std::tie<std::uint64_t,std::uint64_t,threeMotifStructure>
                                                        (node_Outer_key.first,node_Outer_key.second,motifStruc);
                        this_node_possible_motifs->push_back(possible_motif);
                    }
                }
            }
            
            //std::cout<<"Line 606 from process:"<<std::endl;
            return std::move(this_node_possible_motifs);
        };
    
    std::function<threeMotifStructure
                (const DistributedGraph& dg,std::uint64_t node_local_ind,threeMotifStructure para)> 
        evaluate_correct_networkMotifs_oneNode = 
                [](const DistributedGraph& dg,std::uint64_t node_local_ind,threeMotifStructure possible_motif)
        {
            const int my_rank = MPIWrapper::get_my_rank();
            
            if(!(node_local_ind==possible_motif.node_2_local))
                std::cout<<"node_local_ind:"<<node_local_ind<<"   possible_motif.node_2_local:"<<possible_motif.node_2_local<<"   my_rank:"<<my_rank<<"   possible_motif.node_2_rank:"<<possible_motif.node_2_rank<<std::endl;
            assert(node_local_ind==possible_motif.node_2_local);
            assert(my_rank==possible_motif.node_2_rank);
            
            const std::vector<OutEdge>& oEdges = dg.get_out_edges(my_rank,node_local_ind);
            const std::vector<InEdge>& iEdges = dg.get_in_edges(my_rank,node_local_ind);
            
            std::unordered_map<std::pair<std::uint64_t,std::uint64_t>,
                            std::pair<bool,bool>,
                            stdPair_hash> adjacent_nodes;
            
            for(const OutEdge& oEdge : oEdges)
            {
                std::pair<std::uint64_t,std::uint64_t> node_key(oEdge.target_rank,oEdge.target_id);
                std::pair<bool,bool>& value = adjacent_nodes[node_key];
                assert(!value.first);
                value.first = true;
            }
            for(const InEdge& iEdge : iEdges)
            {
                std::pair<std::uint64_t,std::uint64_t> node_key(iEdge.source_rank,iEdge.source_id);
                std::pair<bool,bool>& value = adjacent_nodes[node_key];
                assert(!value.second);
                value.second = true;
            }
            
            std::pair<std::uint64_t,std::uint64_t> node_3_key(possible_motif.node_3_rank,possible_motif.node_3_local);
            std::pair<bool,bool>& value = adjacent_nodes[node_3_key];
            
            bool exists_edge_node2_to_node3 = value.first;
            bool exists_edge_node3_to_node2 = value.second;
            
            if(exists_edge_node2_to_node3 && exists_edge_node3_to_node2)
            // edges between node 2 and 3 in both directions
            {
                //maintain motifs 8,10,11,13
                possible_motif.unsetMotifTypes({1,2,3,4,5,6,7,9,12});
            }
            else if(exists_edge_node2_to_node3 && !exists_edge_node2_to_node3)
            // only edge from node 2 to node 3 
            {
                //maintain motifs 5,7
                possible_motif.unsetMotifTypes({1,2,3,4,6,8,9,10,11,12,13});
            }
            else if(!exists_edge_node2_to_node3 && exists_edge_node2_to_node3)
            // only edge from node 3 to node 2 
            {
                //maintain motifs 12
                possible_motif.unsetMotifTypes({1,2,3,4,5,6,7,8,9,10,11,13});
            }
            else
            // no edges between node 2 and 3
            {
                //maintain motifs 1,2,3,4,6,9
                possible_motif.unsetMotifTypes({5,7,8,10,11,12,13});
            }
            
            assert(possible_motif.checkValidity());
                        
            return possible_motif;
        };
    
    //std::cout<<"Line 674 from process:"<<my_rank<<std::endl;
    
    std::unique_ptr<NodeToNodeQuestionStructure<threeMotifStructure,threeMotifStructure>> threeMotifResults;
    threeMotifResults = std::move(node_to_node_question<threeMotifStructure,threeMotifStructure>
                            (graph,MPIWrapper::MPI_threeMotifStructure,collect_possible_networkMotifs_oneNode,
                                   MPIWrapper::MPI_threeMotifStructure,evaluate_correct_networkMotifs_oneNode));
    
    MPIWrapper::barrier();
    std::cout<<"Line 681 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    //throw std::string("619");


    std::vector<std::uint64_t> motifTypeCount(14,0);
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        std::unique_ptr<std::vector<threeMotifStructure>> this_node_motifs_results;
        this_node_motifs_results = threeMotifResults->getAnswersOfQuestionerNode(node_local_ind);
        
        for(int i=0;i<this_node_motifs_results->size();i++)
        {
            threeMotifStructure& one_motif = (*this_node_motifs_results)[i];
            assert(one_motif.checkValidity());
            for(int motifType=1;motifType<14;motifType++)
            {
                if(one_motif.isMotifTypeSet(motifType))
                    motifTypeCount[motifType]++;
            }
        }
    }
    
    MPIWrapper::barrier();
    std::cout<<"Line 712 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    //throw std::string("712");
    
    std::vector<std::uint64_t> motifTypeCountTotal;
    if(my_rank==resultToRank)
    {
        motifTypeCountTotal.resize(14);
    }
    MPIWrapper::reduce<std::uint64_t>(motifTypeCount.data(),motifTypeCountTotal.data(),                                      
                                      14,MPI_UINT64_T,MPI_SUM,resultToRank);

    MPIWrapper::barrier();
    std::cout<<"Line 726 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    //throw std::string("726");
    
    if(my_rank==resultToRank)
    {
        //Rotational invariant motifs where counted three times each
        assert(motifTypeCountTotal[7]%3==0);
        motifTypeCountTotal[7]/=3;
        assert(motifTypeCountTotal[13]%3==0);
        motifTypeCountTotal[13]/=3;        
    }
    std::vector<long double> motifFraction;
    if(my_rank==resultToRank)
    {
        std::uint64_t total_number_of_motifs = std::accumulate(motifTypeCountTotal.begin(),motifTypeCountTotal.end(),0);
        motifFraction.resize(motifTypeCountTotal.size());
        motifFraction[0] = total_number_of_motifs;
        for(int motifType=1;motifType<14;motifType++)
        {
            motifFraction[motifType] = motifTypeCountTotal[motifType];
            //motifFraction[motifType] = static_cast<long double>(motifTypeCountTotal[motifType]) / static_cast<long double>(total_number_of_motifs);
        }
    }
    MPIWrapper::barrier();
    std::cout<<"Line 751 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    //throw std::string("751");
    return motifFraction;
}

double GraphProperty::computeModularity
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
    std::unique_ptr<std::vector<std::vector<std::string>>> area_names_list_of_ranks = gather_Data_to_all_Ranks<std::string,char>
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
    std::unique_ptr<NodeToNodeQuestionStructure<std::uint64_t,std::uint8_t>> adjacency_results;
    adjacency_results = node_to_node_question<std::uint64_t,std::uint8_t>
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
    for(std::uint64_t node_localID=0;node_localID<number_local_nodes;node_localID++)
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
    for(int area_globalID=0;area_globalID<area_names_list.size();area_globalID++)
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
        
        std::unique_ptr<std::vector<std::vector<NodeModularityInfo>>> local_area_globalID_to_nodeID = gather_Data_to_one_Rank<NodeModularityInfo,NodeModularityInfo>
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

double GraphProperty::computeModularitySingleProc
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

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramm
(
    const DistributedGraph& graph,
    const std::function<std::unique_ptr<Histogram>(const double, const double)> histogram_creator,
    const unsigned int resultToRank
)
{
// Test function parameters
    const int my_rank = MPIWrapper::get_my_rank();
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    if(resultToRank>=number_of_ranks)
    {
        throw std::invalid_argument("Bad parameter - resultToRank:"+resultToRank);
    }
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
        
//Compute node local edge lengths of OutEdges
    std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,Vec3d>>>
                 (const DistributedGraph& dg,std::uint64_t node_local_ind)>
        transfer_node_position = [&](const DistributedGraph& dg,std::uint64_t node_local_ind)
        {
            const std::vector<OutEdge>& oEdges = dg.get_out_edges(my_rank,node_local_ind);
            Vec3d source_node_pos = dg.get_node_position(my_rank,node_local_ind);
            auto node_position_vec =
                std::make_unique<std::vector<std::tuple<std::uint64_t,std::uint64_t,Vec3d>>>(oEdges.size());
            for(int i=0;i<oEdges.size();i++)
            {
                (*node_position_vec)[i] = std::tie(oEdges[i].target_rank,oEdges[i].target_id,source_node_pos);
            }
            return std::move(node_position_vec);
        };
    std::function<double(const DistributedGraph& dg,std::uint64_t node_local_ind,Vec3d para)> 
        compute_edge_length = [&](const DistributedGraph& dg,std::uint64_t node_local_ind,Vec3d source_node_pos)
        {
            Vec3d target_node_pos = dg.get_node_position(my_rank,node_local_ind);
            return (source_node_pos-target_node_pos).calculate_p_norm(2);
        };
    std::unique_ptr<NodeToNodeQuestionStructure<Vec3d,double>> edge_length_results=
        node_to_node_question<Vec3d,double>(graph,MPIWrapper::MPI_Vec3d,transfer_node_position,
                                                  MPI_DOUBLE,compute_edge_length);

// Collect edge lengths of edges to local list
    std::vector<double> edge_lengths;    
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        std::unique_ptr<std::vector<double>> length_of_all_edges=
            edge_length_results->getAnswersOfQuestionerNode(node_local_ind);
        edge_lengths.insert(edge_lengths.end(),length_of_all_edges->begin(),length_of_all_edges->end());
    }
    
// Compute the smallest and largest edge length globally
    const auto [min_length, max_length] = std::minmax_element(edge_lengths.begin(), edge_lengths.end());
    double global_min_length = *min_length;
    double global_max_length = *max_length;    
    global_min_length = MPIWrapper::all_reduce<double>(global_min_length,MPI_DOUBLE,MPI_MIN);
    global_max_length = MPIWrapper::all_reduce<double>(global_max_length,MPI_DOUBLE,MPI_MAX);
    
//Create histogram with local edge data
    std::unique_ptr<Histogram> histogram = histogram_creator(global_min_length,global_max_length);
    std::pair<double,double> span = histogram->front().first;
    double bin_width = span.second - span.first;
    assert(bin_width>0);
    double start_length = span.first;
    for(const double length : edge_lengths)
    {
        int index = (length-start_length)/bin_width;
        assert(index<histogram->size());
        assert(index>=0);
        assert(length>=(*histogram)[index].first.first);
        assert(length<(*histogram)[index].first.second);
        (*histogram)[index].second++;
    }
    
// Reduce local edge count of histogram to global count        
    std::vector<std::uint64_t> histogram_pure_count_src(histogram->size());
    for(int i=0;i<histogram->size();i++)
    {
        histogram_pure_count_src[i] = (*histogram)[i].second;
    }
    std::vector<std::uint64_t> histogram_pure_count_dest;
    if(my_rank==resultToRank)
    {
        histogram_pure_count_dest.resize(histogram->size());
    }    
    MPIWrapper::reduce<std::uint64_t>(histogram_pure_count_src.data(),histogram_pure_count_dest.data(),
                                      histogram->size(),MPI_UINT64_T,MPI_SUM,resultToRank);
    
// Reconstruct resulting histogram with global data
    if(my_rank==resultToRank)
    {
        for(int i=0;i<histogram->size();i++)
        {
            (*histogram)[i].second = histogram_pure_count_dest[i];
        }
    }
    else
    {
        histogram = std::make_unique<std::vector<std::pair<std::pair<double,double>,std::uint64_t>>>();
    }
    
    return std::move(histogram);
}

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramSingleProc
(
    const DistributedGraph& graph,
    std::function<std::unique_ptr<Histogram>(double, double)> histogram_creator,
    unsigned int resultToRank
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const int number_ranks = MPIWrapper::get_number_ranks();

    // Main rank gathers other ranks number of nodes
    std::vector<std::uint64_t> number_nodes_of_ranks;
    number_nodes_of_ranks.resize(number_ranks);
    std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    MPIWrapper::gather<uint64_t>(&number_local_nodes, number_nodes_of_ranks.data(), 1, MPI_UINT64_T, resultToRank);
    auto histogram = std::make_unique<std::vector<std::pair<std::pair<double,double>,std::uint64_t>>>();

    // Only the main rank collects all edge lengths and sorts each length into the 
    // buckets of the histogram created using the supplied function
    if(my_rank == resultToRank)
    {   
        std::vector<double> edge_lengths;

        double max_length = 0;      // assumption: min edge length is 0
        double min_length = std::numeric_limits<double>::max();    // assumption: max edge length is 300

        // Iterate through each rank...
        for (int rank = 0; rank < number_ranks; rank++)
        {
            // ...and each node of a rank...
            for(int node = 0; node < number_nodes_of_ranks[rank]; node++)
            {
                // ...and compute the distance of each outgoing edge of a node 
                const Vec3d node_position = graph.get_node_position(rank, node);
                const std::vector<OutEdge>& out_edges = graph.get_out_edges(rank, node);

                for (const auto& [target_rank, target_id, weight] : out_edges)
                {
                    const Vec3d target_position = graph.get_node_position(target_rank, target_id);

                    const Vec3d difference = target_position - node_position;
                    const double length = difference.calculate_2_norm();
                    edge_lengths.push_back(length);

                    // Find out bounds of the edge lengths for the histogram creation function
                    if(length > max_length) max_length = length;
                    if(length < min_length) min_length = length;
                }
            }
        }
        std::cout << "total edge count: " << edge_lengths.size() << std::endl;
        histogram = histogram_creator(min_length, max_length);
        
        // For each bucket iterate through the edge lengths vector and increase bucket counter if corresponding edge length is found
        for (int i = 0; i < histogram->size(); i++)
        {
            for (int j = 0; j < edge_lengths.size(); j++)
            {
                // Count edge lengths inside interval greater equal the lower and smaller than the upper bound
                if(edge_lengths[j] >= (*histogram)[i].first.first &&
                   edge_lengths[j] < (*histogram)[i].first.second)
                {
                    (*histogram)[i].second++;
                }
            }  
        }
        // Print out histogram
        for (int i = 0; i < histogram->size(); i++)
        {
            std::cout << i << ". " << "bin: " << (*histogram)[i].first.first << "-" << (*histogram)[i].first.second << ": " << (*histogram)[i].second << std::endl;
        }
    }
    return std::move(histogram);
}

template<typename Q_parameter,typename A_parameter>
std::unique_ptr<GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>> GraphProperty::node_to_node_question
(
    const DistributedGraph& graph,
    MPI_Datatype MPI_Q_parameter,
    std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,Q_parameter>>>(const DistributedGraph& dg,std::uint64_t node_local_ind)> generateAddressees,
    MPI_Datatype MPI_A_parameter,
    std::function<A_parameter(const DistributedGraph& dg,std::uint64_t node_local_ind,Q_parameter para)> generateAnswers
)
{
    std::vector<double> times;
    std::vector<std::string> code_names = {"GenQuestion","DistrNumbers","DistrQuestions","SetQuestions","CompAnswers",
                                            "SendAnswers","SetAnswers"};
    auto start = std::chrono::steady_clock::now(); 

    const int my_rank = MPIWrapper::get_my_rank();
    const int number_ranks = MPIWrapper::get_number_ranks();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    
// Collect Questions and create Questioners structure
    auto questioner_structure = std::make_unique<GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>>();
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        questioner_structure->addQuestionsFromOneNodeToSend(generateAddressees(graph,node_local_ind),node_local_ind);
    }
    questioner_structure->finalizeAddingQuestionsToSend();
    
    times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
    start = std::chrono::steady_clock::now(); 
            
// Distribute number of questions to each rank
    std::vector<int>& send_ranks_to_nbrOfQuestions = questioner_structure->get_adressee_ranks_to_nbrOfQuestions();
    std::vector<int> global_ranks_to_nbrOfQuestions(number_ranks*number_ranks);    
    std::vector<int> destCounts_ranks_to_nbrOfQuestions(number_ranks,number_ranks);
    std::vector<int> displ_ranks_to_nbrOfQuestions(number_ranks);
    for(int index = 0;index<displ_ranks_to_nbrOfQuestions.size();index++)
    {
        displ_ranks_to_nbrOfQuestions[index]=number_ranks*index;
    }
    MPIWrapper::all_gatherv<int>(send_ranks_to_nbrOfQuestions.data(), number_ranks,
                                global_ranks_to_nbrOfQuestions.data(), destCounts_ranks_to_nbrOfQuestions.data(), displ_ranks_to_nbrOfQuestions.data(), MPI_INT);

    times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
    start = std::chrono::steady_clock::now(); 
    
// Distribute questions to each rank
    std::vector<int> recv_ranks_to_nbrOfQuestions(number_ranks);
    for(int rank=0;rank<recv_ranks_to_nbrOfQuestions.size();rank++)
    {
        recv_ranks_to_nbrOfQuestions[rank] = global_ranks_to_nbrOfQuestions[rank*number_ranks+my_rank];
    }
    std::vector<int> displ_recv_ranks_to_nbrOfQuestions(number_ranks,0);
    for(int index = 1;index<displ_recv_ranks_to_nbrOfQuestions.size();index++)
    {
        displ_recv_ranks_to_nbrOfQuestions[index]=
            displ_recv_ranks_to_nbrOfQuestions[index-1]+recv_ranks_to_nbrOfQuestions[index-1];
    }
    int my_rank_total_receive_size = std::accumulate(recv_ranks_to_nbrOfQuestions.begin(),
                                                     recv_ranks_to_nbrOfQuestions.end(),0);
    std::vector<std::uint64_t> my_rank_total_nodes_to_ask_question(my_rank_total_receive_size);
    std::vector<Q_parameter> my_rank_total_question_parameters(my_rank_total_receive_size);
    for(int rank=0;rank<number_ranks;rank++)
    {
        std::vector<std::uint64_t>& nodes_to_ask_question_for_rank =
            questioner_structure->get_nodes_to_ask_question_for_rank(rank);
        std::vector<Q_parameter>& question_parameters_for_rank =
            questioner_structure->get_question_parameters_for_rank(rank);
        assert(nodes_to_ask_question_for_rank.size()==question_parameters_for_rank.size());
        int count = nodes_to_ask_question_for_rank.size();
        
        MPIWrapper::gatherv<std::uint64_t>(nodes_to_ask_question_for_rank.data(), count,
                                           my_rank_total_nodes_to_ask_question.data(),
                                           recv_ranks_to_nbrOfQuestions.data(), displ_recv_ranks_to_nbrOfQuestions.data(),MPI_UINT64_T,rank);
        
        MPIWrapper::gatherv<Q_parameter>(question_parameters_for_rank.data(), count,
                                         my_rank_total_question_parameters.data(),
                                         recv_ranks_to_nbrOfQuestions.data(), displ_recv_ranks_to_nbrOfQuestions.data(),MPI_Q_parameter,rank);
    }
    
    times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
    start = std::chrono::steady_clock::now(); 

// Set questions to be answered to adressees questioner structure
    GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter> adressee_structure;
    adressee_structure.setQuestionsReceived(my_rank_total_nodes_to_ask_question,my_rank_total_question_parameters,
                                            recv_ranks_to_nbrOfQuestions,displ_recv_ranks_to_nbrOfQuestions);
    
    times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
    start = std::chrono::steady_clock::now(); 
    
// Compute the answers to questions
    adressee_structure.computeAnswersToQuestions(graph,generateAnswers);
    
    times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
    start = std::chrono::steady_clock::now(); 
    
// Send answers back to questioners
    std::vector<int>& send_ranks_to_nbrOfAnswers =  questioner_structure->get_adressee_ranks_to_nbrOfQuestions();
    std::vector<int> displ_send_ranks_to_nbrOfAnswers(number_ranks,0);
    for(int index = 1;index<displ_send_ranks_to_nbrOfAnswers.size();index++)
    {
        displ_send_ranks_to_nbrOfAnswers[index]=
            displ_send_ranks_to_nbrOfAnswers[index-1]+send_ranks_to_nbrOfAnswers[index-1];
    }
    my_rank_total_receive_size = std::accumulate(send_ranks_to_nbrOfAnswers.begin(),
                                                 send_ranks_to_nbrOfAnswers.end(),0);
    std::vector<A_parameter> my_rank_total_answer_parameters(my_rank_total_receive_size);
    for(int rank=0;rank<number_ranks;rank++)
    {
        std::vector<A_parameter>& answers_for_rank = adressee_structure.get_answers_for_rank(rank);
        
        int count = answers_for_rank.size();        
        
        A_parameter* intermed = answers_for_rank.data();
        
        MPIWrapper::gatherv<A_parameter>(intermed, count,
                                         my_rank_total_answer_parameters.data(),
                                         send_ranks_to_nbrOfAnswers.data(), displ_send_ranks_to_nbrOfAnswers.data(),MPI_A_parameter,rank);
    }
    
    times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
    start = std::chrono::steady_clock::now(); 
    
// Set answers questioner structure
    questioner_structure->setAnswers(my_rank_total_answer_parameters,send_ranks_to_nbrOfAnswers,
                                    displ_send_ranks_to_nbrOfAnswers);

    times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
    start = std::chrono::steady_clock::now(); 
    
    std::cout<<"Rank:"<<my_rank<<"[ ";
    for(int i=0;i<times.size();i++)
    {
        std::cout<<" "<<code_names[i]<<":"<<times[i]<<" ";
    }
    std::cout<<"]"<<std::endl;
    
    return std::move(questioner_structure);
}

template<typename Q_parameter,typename A_parameter>
void GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>::addQuestionsFromOneNodeToSend
(
    std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,Q_parameter>>> list_of_adressees_and_parameter,
    std::uint64_t questioner
)
{
    assert(structureStatus==Empty || structureStatus==PrepareQuestionsToSend);
        
    /*
    MPIWrapper::barrier();
    std::cout<<"Line 1203 from process:------------------------------------"<<MPIWrapper::get_my_rank()<<std::endl;
    fflush(stdout);
    std::cout<<"list_of_adressees_and_parameter.size():"<<list_of_adressees_and_parameter.size()<<std::endl;
    std::cout<<"questioner:"<<questioner<<std::endl;
    MPIWrapper::barrier();
    */
    
    for(auto [target_rank,target_local_node,Q_parameter_struct] : *list_of_adressees_and_parameter)
    {
        assert(target_rank<MPIWrapper::get_number_ranks());
        const auto rank_to_index = rank_to_outerIndex.find(target_rank);
        //std::cout<<"target_rank:"<<target_rank<<"  target_local_node:"<<target_local_node<<std::endl;
        if(rank_to_index != rank_to_outerIndex.end())
        // If rank was already encoutered due to other outEdge 
        {
            std::uint64_t outerIndex = rank_to_index->second;
            /*
            if(!(outerIndex<nodes_to_ask_question.size() && outerIndex>=0))
                std::cout<<outerIndex<<"-----------------------------------------------------------"<<nodes_to_ask_question.size()<<std::endl;
            */
            
            assert(outerIndex>=0);
            assert(outerIndex<nodes_to_ask_question.size());
            assert(outerIndex<nodes_that_ask_the_question.size());
            assert(outerIndex<question_parameters.size());
            
            assert(nodes_that_ask_the_question[outerIndex].size()==
                   nodes_to_ask_question[outerIndex].size());
            assert(nodes_that_ask_the_question[outerIndex].size()==
                   question_parameters[outerIndex].size());
            
            std::uint64_t innerIndex = nodes_to_ask_question[outerIndex].size();
                        
            nodes_to_ask_question[outerIndex].push_back(target_local_node);
            nodes_that_ask_the_question[outerIndex].push_back(questioner);
            question_parameters[outerIndex].push_back(Q_parameter_struct);
                        
            auto doublePair = 
            std::pair<std::uint64_t,std::pair<std::uint64_t,std::uint64_t>>(questioner,{outerIndex,innerIndex});
            questioner_node_to_outerIndex_and_innerIndex.insert(doublePair);
        }
        else
        {
            std::uint64_t outerIndex = nodes_to_ask_question.size();
            std::uint64_t innerIndex = 0;
            
            nodes_to_ask_question.push_back({target_local_node});
            nodes_that_ask_the_question.push_back({questioner});
            question_parameters.push_back({Q_parameter_struct});
            
            auto doublePair = 
            std::pair<std::uint64_t,std::pair<std::uint64_t,std::uint64_t>>(questioner,{outerIndex,innerIndex});
            questioner_node_to_outerIndex_and_innerIndex.insert(doublePair);
                    
            rank_to_outerIndex.insert({target_rank,outerIndex});
        }
    }
    
    for(auto keyValue = questioner_node_to_outerIndex_and_innerIndex.begin();
        keyValue!=questioner_node_to_outerIndex_and_innerIndex.end();
        keyValue++)
    {
        std::uint64_t questioner = keyValue->first;
        std::uint64_t outerIndex = keyValue->second.first;
        assert(outerIndex>=0);
        assert(outerIndex<nodes_to_ask_question.size());
        assert(outerIndex<nodes_that_ask_the_question.size());
        assert(outerIndex<question_parameters.size());
        std::uint64_t innerIndex = keyValue->second.second;
        assert(innerIndex>=0);
        assert(innerIndex<nodes_to_ask_question[outerIndex].size());
        assert(innerIndex<nodes_that_ask_the_question[outerIndex].size());
        assert(questioner==nodes_that_ask_the_question[outerIndex][innerIndex]);
        assert(innerIndex<question_parameters[outerIndex].size());
    }
    
    structureStatus = PrepareQuestionsToSend;
    /*
    MPIWrapper::barrier();
    std::cout<<"Line 1245 from process:------------------------------------"<<MPIWrapper::get_my_rank()<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    */
}

template<typename Q_parameter,typename A_parameter>
void GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>::setQuestionsReceived
(
    std::vector<std::uint64_t>& total_nodes_to_ask_question,
    std::vector<Q_parameter>& total_question_parameters,
    std::vector<int>& rank_size,
    std::vector<int>& rank_displ
)
{
    assert(rank_size.size()==rank_displ.size());
    for(int rank=0;rank<rank_size.size();rank++)
    {
        int nbr_of_questions = rank_size[rank];
        if(nbr_of_questions > 0)
        {
            int outerIndex = list_index_to_adressee_rank.size();
            rank_to_outerIndex.insert(std::pair<std::uint64_t,std::uint64_t>(rank,outerIndex));
            
            assert(outerIndex==nodes_to_ask_question.size());
            assert(rank_displ[rank]<total_nodes_to_ask_question.size() && rank_displ[rank]>=0);
            
            list_index_to_adressee_rank.push_back(rank);
            nodes_to_ask_question.push_back({});
            nodes_to_ask_question.back().resize(nbr_of_questions);
            assert(nodes_to_ask_question.back().size()==nbr_of_questions);
            assert(rank_displ[rank]>=0 && rank_displ[rank]<total_nodes_to_ask_question.size());
            std::memcpy(nodes_to_ask_question.back().data(),&total_nodes_to_ask_question[rank_displ[rank]],                        nbr_of_questions*sizeof(std::uint64_t));
            
            question_parameters.push_back({});
            question_parameters.back().resize(nbr_of_questions);
            assert(outerIndex<question_parameters.size());
            assert(rank_displ[rank]<total_question_parameters.size() && rank_displ[rank]>=0);
            assert(nbr_of_questions<=question_parameters[outerIndex].size());
            std::memcpy(question_parameters.back().data(),&total_question_parameters[rank_displ[rank]],nbr_of_questions*sizeof(Q_parameter));
            
            addressee_ranks_to_nbrOfQuestions.push_back(nbr_of_questions);
        }
    }
    //throw std::string("1544");
}

template<typename Q_parameter,typename A_parameter>
void GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>::setAnswers
(
    std::vector<A_parameter>& total_answers,
    std::vector<int>& rank_size,
    std::vector<int>& rank_displ
)
{
    int number_answer_blocks = 0;
    for(int rank=0;rank<rank_size.size();rank++)
    {
        if(rank_size[rank]>0)
        {
            number_answer_blocks++;
        }
    }
    assert(number_answer_blocks==nodes_to_ask_question.size());
    assert(number_answer_blocks==list_index_to_adressee_rank.size());
    assert(number_answer_blocks==nodes_that_ask_the_question.size());
    assert(number_answer_blocks==question_parameters.size());
    
    answers_to_questions.resize(number_answer_blocks);
    for(int rank=0;rank<rank_size.size();rank++)
    {
        int nbr_of_answers = rank_size[rank];
        if(nbr_of_answers > 0)
        {
            auto keyValue = rank_to_outerIndex.find(rank);
            assert(keyValue!=rank_to_outerIndex.end());
            int outerIndex = keyValue->second;
            assert(outerIndex<answers_to_questions.size() && outerIndex>=0);
            assert(answers_to_questions[outerIndex].size()==0);
            assert(nbr_of_answers==nodes_to_ask_question[outerIndex].size());
            assert(nbr_of_answers==nodes_that_ask_the_question[outerIndex].size());
            assert(nbr_of_answers==question_parameters[outerIndex].size());            answers_to_questions[outerIndex].resize(nbr_of_answers);
            
            /*
            for(int i=0;i<nbr_of_answers;i++)
            {
                answers_to_questions[outerIndex][i] = total_answers[rank_displ[rank]+i];
            }
            */
            std::memcpy(answers_to_questions[outerIndex].data(),&total_answers[rank_displ[rank]],nbr_of_answers*sizeof(A_parameter));
        }
    }
}

template<typename Q_parameter,typename A_parameter>
void GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>::computeAnswersToQuestions
(
    const DistributedGraph& dg,
    std::function<A_parameter(const DistributedGraph& dg,std::uint64_t node_local_ind,Q_parameter para)> generateAnswers
)
{
    assert(nodes_to_ask_question.size()==question_parameters.size());
    answers_to_questions.resize(question_parameters.size());
    for(int i=0; i<question_parameters.size(); i++)
    {
        assert(nodes_to_ask_question[i].size()==question_parameters[i].size());
        answers_to_questions[i].resize(question_parameters[i].size());
        for(int j=0; j<question_parameters[i].size(); j++)
        {
            std::uint64_t node_local_ind = nodes_to_ask_question[i][j];
            Q_parameter para = question_parameters[i][j];
            answers_to_questions[i][j] = generateAnswers(dg,node_local_ind,para);
        }            
        //std::cout<<"Rank:"<<MPIWrapper::get_my_rank()<<"  "<<i<<"/"<<question_parameters.size()<<"  Generated answer for:"<<answers_to_questions[i].size()<<std::endl; 
    }
}

template<typename Q_parameter,typename A_parameter>
std::vector<int>& GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>::get_adressee_ranks_to_nbrOfQuestions()
{
    assert(structureStatus==ClosedQuestionsPreparation);
    return addressee_ranks_to_nbrOfQuestions;
}

template<typename Q_parameter,typename A_parameter>
std::vector<std::uint64_t>& GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>::get_nodes_to_ask_question_for_rank
(
    std::uint64_t ranks
)
{
    auto keyValue = rank_to_outerIndex.find(ranks);
    if(keyValue!=rank_to_outerIndex.end() && nodes_to_ask_question.size()!=0)
    {
        std::uint64_t outerIndex = keyValue->second;
        assert(outerIndex<nodes_to_ask_question.size());
        return nodes_to_ask_question[outerIndex];
    }
    return dummy_nodes_to_ask_question;
}

template<typename Q_parameter,typename A_parameter>
std::vector<Q_parameter>& GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>::get_question_parameters_for_rank
(
    std::uint64_t ranks
)
{
    auto keyValue = rank_to_outerIndex.find(ranks);
    if(keyValue!=rank_to_outerIndex.end() && question_parameters.size()!=0)
    {
        std::uint64_t outerIndex = keyValue->second;
        assert(outerIndex<question_parameters.size());
        return question_parameters[outerIndex];
    }
    return dummy_question_parameters;
}

template<typename Q_parameter,typename A_parameter>
std::vector<A_parameter>& GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>::get_answers_for_rank
(
    std::uint64_t ranks
)
{
    auto keyValue = rank_to_outerIndex.find(ranks);
    if(keyValue!=rank_to_outerIndex.end() && answers_to_questions.size()!=0)
    {
        std::uint64_t outerIndex = keyValue->second;
        assert(outerIndex<answers_to_questions.size());
        return answers_to_questions[outerIndex];
    }
    else
    {
        return dummy_answers_to_questions;
    }
}

template<typename Q_parameter,typename A_parameter>
std::unique_ptr<std::vector<A_parameter>> GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>::getAnswersOfQuestionerNode
(
    std::uint64_t node_local_ind
)
{
    auto answers = std::make_unique<std::vector<A_parameter>>();
    for(auto keyValue = questioner_node_to_outerIndex_and_innerIndex.find(node_local_ind);
        keyValue!=questioner_node_to_outerIndex_and_innerIndex.end() && keyValue->first == node_local_ind;
        keyValue++)
    {
        
        std::uint64_t outerIndex = keyValue->second.first;
        assert(outerIndex<answers_to_questions.size() && outerIndex>=0);
        std::uint64_t innerIndex = keyValue->second.second;
        if(!(innerIndex<answers_to_questions[outerIndex].size() && innerIndex>=0))
            std::cout<<"-------"<<outerIndex<<"---------"<<innerIndex<<"---------"<<answers_to_questions[outerIndex].size()<<std::endl;
        assert(innerIndex<answers_to_questions[outerIndex].size() && innerIndex>=0);
        answers->push_back(answers_to_questions[outerIndex][innerIndex]);
    }
    return std::move(answers);
}

template<typename Q_parameter,typename A_parameter>
void GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>::finalizeAddingQuestionsToSend()
{
    int number_ranks = MPIWrapper::get_number_ranks();
    assert(structureStatus==PrepareQuestionsToSend);
    assert(nodes_to_ask_question.size()==nodes_that_ask_the_question.size());
    assert(nodes_that_ask_the_question.size()==question_parameters.size());
    
    list_index_to_adressee_rank.resize(nodes_to_ask_question.size(),-1);
    for(auto iter=rank_to_outerIndex.begin(); iter!=rank_to_outerIndex.end(); iter++)
    {
        assert(iter->first<number_ranks && iter->first>=0);
        assert(iter->second<nodes_to_ask_question.size() && iter->second>=0);
        assert(list_index_to_adressee_rank[iter->second]==-1);
        list_index_to_adressee_rank[iter->second] = iter->first;
    }

    /*
    for(std::vector<std::uint64_t>& list_nbrToRank: nodes_to_ask_question)
    {
        std::cout<<"nodes_to_ask_question: "<<list_nbrToRank.size();
        std::cout<<std::endl;
    }
    for(std::uint64_t list_nbrToRank: list_index_to_adressee_rank)
    {
        std::cout<<"list_index_to_adressee_rank: "<<list_nbrToRank;
        std::cout<<std::endl;
    }
    */
    
    addressee_ranks_to_nbrOfQuestions.resize(number_ranks,0);
    for(int index=0;index<list_index_to_adressee_rank.size();index++)
    {
        std::uint64_t rank = list_index_to_adressee_rank[index];
        assert(rank<number_ranks);
        addressee_ranks_to_nbrOfQuestions[rank] = nodes_to_ask_question[index].size();
    }

    /*
    for(std::uint64_t list_nbrToRank: addressee_ranks_to_nbrOfQuestions)
    {
        std::cout<<"addressee_ranks_to_nbrOfQuestions: "<<list_nbrToRank<<std::endl;
    }
    */
    
    structureStatus = ClosedQuestionsPreparation;
    
    for(auto keyValue = questioner_node_to_outerIndex_and_innerIndex.begin();
        keyValue!=questioner_node_to_outerIndex_and_innerIndex.end();
        keyValue++)
    {
        std::uint64_t questioner = keyValue->first;
        std::uint64_t outerIndex = keyValue->second.first;
        assert(outerIndex>=0);
        assert(outerIndex<nodes_to_ask_question.size());
        assert(outerIndex<list_index_to_adressee_rank.size());
        assert(outerIndex<nodes_that_ask_the_question.size());
        assert(outerIndex<question_parameters.size());
        std::uint64_t innerIndex = keyValue->second.second;
        assert(innerIndex>=0);
        assert(innerIndex<nodes_to_ask_question[outerIndex].size());
        assert(innerIndex<nodes_that_ask_the_question[outerIndex].size());
        assert(questioner==nodes_that_ask_the_question[outerIndex][innerIndex]);
        assert(innerIndex<question_parameters[outerIndex].size());
    }
    
    
    // Testing
    /*
    for(int index=0; index<list_index_to_adressee_rank.size(); index++)
    {
        int targetRank = list_index_to_adressee_rank[index];
        for(int j=0;j<nodes_to_ask_question[index].size();j++)
        {
            int targetNode = nodes_to_ask_question[index][j];
            int sourceNode = nodes_that_ask_the_question[index][j];
            threeMotifStructure struc = question_parameters[index][j];
            assert(struc.node_1_rank==MPIWrapper::get_my_rank());
            assert(struc.node_1_local==sourceNode);
            assert(struc.node_2_rank==targetRank);
            assert(struc.node_2_local==targetNode);
        }
        auto keyValue = rank_to_outerIndex.find(targetRank);
        assert(keyValue!=rank_to_outerIndex.end());
        std::uint64_t outerIndex = keyValue->second;
        assert(outerIndex==index);
    }
    */
}

template<typename DATA,typename DATA_Element>
std::unique_ptr<std::vector<std::vector<DATA>>> GraphProperty::gather_Data_to_one_Rank
(
    const DistributedGraph& dg,
    std::function<std::unique_ptr<std::vector<std::pair<DATA,int>>>(const DistributedGraph&)> getData,
    std::function<std::vector<DATA_Element>(DATA)> transformDataToElementary,
    std::function<DATA(std::vector<DATA_Element>&)> transformElementaryToData,
    MPI_Datatype DATA_Element_datatype,
    int root
)
{
    auto dataGatherMethod=[](DATA_Element* src, int count, DATA_Element* dest, int* destCounts,
                             int* displs, MPI_Datatype datatype,int root)
    {
        MPIWrapper::gatherv<DATA_Element>(src,count,dest,destCounts,displs,datatype,root);
    };
    
    auto sizesGatherMethod=[](int* src, int count, int* dest, int* destCounts,
                              int* displs, int root)
    {
        MPIWrapper::gatherv<int>(src,count,dest,destCounts,displs,MPI_INT,root);
    };

    return std::move(gather_Data<DATA,DATA_Element>(
        dg,getData,transformDataToElementary,transformElementaryToData,
        dataGatherMethod,sizesGatherMethod,DATA_Element_datatype,root
    ));
}

template<typename DATA,typename DATA_Element>
std::unique_ptr<std::vector<std::vector<DATA>>> GraphProperty::gather_Data_to_all_Ranks
(
    const DistributedGraph& dg,
    std::function<std::unique_ptr<std::vector<std::pair<DATA,int>>>(const DistributedGraph&)> getData,
    std::function<std::vector<DATA_Element>(DATA)> transformDataToElementary,
    std::function<DATA(std::vector<DATA_Element>&)> transformElementaryToData,
    MPI_Datatype DATA_Element_datatype
)
{ 
    auto dataGatherMethod=[](DATA_Element* src, int count, DATA_Element* dest, int* destCounts,
                             int* displs, MPI_Datatype datatype,int root)
    {
        MPIWrapper::all_gatherv<DATA_Element>(src,count,dest,destCounts,displs,datatype);
    };
    
    auto sizesGatherMethod=[](int* src, int count, int* dest, int* destCounts,
                              int* displs, int root)
    {
        MPIWrapper::all_gatherv<int>(src,count,dest,destCounts,displs,MPI_INT);
    };

    return std::move(gather_Data<DATA,DATA_Element>(
        dg,getData,transformDataToElementary,transformElementaryToData,dataGatherMethod,
        sizesGatherMethod,DATA_Element_datatype,-1)
    );
}

template<typename DATA,typename DATA_Element>
std::unique_ptr<std::vector<std::vector<DATA>>> GraphProperty::gather_Data
(
    const DistributedGraph& dg,
    std::function<std::unique_ptr<std::vector<std::pair<DATA,int>>>(const DistributedGraph& dg)> getData,
    std::function<std::vector<DATA_Element>(DATA dat)> transformDataToElementary,
    std::function<DATA(std::vector<DATA_Element>&)> transformElementaryToData,
    std::function<void(DATA_Element* src, int count, DATA_Element* dest, int* destCounts, int* displs, MPI_Datatype datatype,int root)> dataGatherMethod,
    std::function<void(int* src, int count, int* dest, int* destCounts, int* displs,int root)> sizesGatherMethod,
    MPI_Datatype DATA_Element_datatype,
    int root
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const int number_ranks = MPIWrapper::get_number_ranks();
    int data_target_size = (root==-1 || root==my_rank)?number_ranks:0;
    
    //Generate and transform data
    std::unique_ptr<std::vector<std::pair<DATA,int>>> data = getData(dg);
    int local_number_data = data->size();
    std::vector<int> data_inner_size(local_number_data);
    std::transform(data->begin(),data->end(),data_inner_size.begin(),
                   [](std::pair<DATA,int> p){return p.second;});
    std::vector<DATA_Element> local_DATA_Elements;
    std::for_each(data->cbegin(),data->cend(),
                  [&](std::pair<DATA,int> p)
                    { 
                        std::vector<DATA_Element> vec = transformDataToElementary(p.first);
                        local_DATA_Elements.insert(local_DATA_Elements.end(),vec.begin(),vec.end());
                    });
    int local_DATA_Elements_size = local_DATA_Elements.size();
    
    /*
    MPIWrapper::barrier();
    std::cout<<"----------------------------------------------------------------------"<<std::endl;
    MPIWrapper::barrier();
    
    if(my_rank==2)
    {
        std::cout<<"Rank:"<<my_rank<<"  size:"<<data_inner_size.size()<<std::endl;
        for(int i=0;i<data_inner_size.size();i++)
                std::cout<<" "<<data_inner_size[i];
        std::cout<<"Rank:"<<my_rank<<"  size:"<<local_DATA_Elements.size()<<std::endl;
        for(int i=0;i<local_DATA_Elements.size();i++)
                std::cout<<" "<<local_DATA_Elements[i]; 
        std::cout<<std::endl;    
        fflush(stdout);
    }
    MPIWrapper::barrier();
    std::cout<<"----------------------------------------------------------------------"<<std::endl;
    MPIWrapper::barrier();
    */
    
    //Gather number of DATA items
    std::vector<int> global_local_number_data(data_target_size);
    std::vector<int> destCountNbr(data_target_size,1);
    std::vector<int> displsNbr(data_target_size,0);
    if(root==-1 || root==my_rank)
    {
        std::partial_sum(destCountNbr.begin(), destCountNbr.end()-1,
                         displsNbr.begin()+1, std::plus<int>());
    }
    sizesGatherMethod(&local_number_data,1,global_local_number_data.data(),destCountNbr.data(),
                      displsNbr.data(),root);
    
    //Gather inner size of data elements
    std::vector<int> global_data_inner_size;
    std::vector<int> destCountInnerNbr(data_target_size);
    std::vector<int> displsInnerNbr(data_target_size,0);
    if(root==-1 || root==my_rank)
    {
        global_data_inner_size.resize(std::accumulate(global_local_number_data.cbegin(),
                                                      global_local_number_data.cend(),0));
        std::transform(global_local_number_data.begin(),global_local_number_data.end(),
                       destCountInnerNbr.begin(),[](int len){return len;});
        std::partial_sum(destCountInnerNbr.begin(), destCountInnerNbr.end()-1,
                         displsInnerNbr.begin()+1, std::plus<int>());
    }
    sizesGatherMethod(data_inner_size.data(),local_number_data,global_data_inner_size.data(),
                      destCountInnerNbr.data(),displsInnerNbr.data(),root);
    
    //Gather DATA_Elements
    std::vector<DATA_Element> global_DATA_Elements;
    std::vector<int> destCountDATAElements(data_target_size);
    std::vector<int> displsInnerDATAElements(data_target_size,0);
    if(root==-1 || root==my_rank)
    {
        global_DATA_Elements.resize(std::accumulate(global_data_inner_size.begin(),
                                                    global_data_inner_size.end(),0));
        int index=0;
        std::transform(global_local_number_data.begin(),global_local_number_data.end(),
                       destCountDATAElements.begin(),[&](int len)
                        { 
                            int count = std::accumulate(global_data_inner_size.begin()+index,global_data_inner_size.begin()+index+len,0);
                            index+=len;
                            return count;
                        });
        std::partial_sum(destCountDATAElements.begin(),destCountDATAElements.end()-1,
                         displsInnerDATAElements.begin()+1,std::plus<int>());
    }
    dataGatherMethod(local_DATA_Elements.data(),local_DATA_Elements_size,
                     global_DATA_Elements.data(),destCountDATAElements.data(),
                     displsInnerDATAElements.data(),DATA_Element_datatype,root);

    /*
    MPIWrapper::barrier();
    if(my_rank==0)
    {
        std::cout<<"Rank:"<<my_rank<<"  size:"<<destCountDATAElements[2]<<std::endl;
        for(int i=displsInnerDATAElements[2];i<displsInnerDATAElements[2]+destCountDATAElements[2];i++)
                std::cout<<" "<<global_DATA_Elements[i]; 
        std::cout<<std::endl;    
        fflush(stdout);
    }
    
    MPIWrapper::barrier();
    std::cout<<"----------------------------------------------------------------------"<<std::endl;
    MPIWrapper::barrier();
    */
    
    //Reorganize DATA
    auto collectedData = std::make_unique<std::vector<std::vector<DATA>>>();
    if(root==-1 || root==my_rank)
    {
        collectedData->resize(number_ranks);
        for(int rank=0;rank<number_ranks;rank++)
        {
            int rank_DataElement_Start = displsInnerDATAElements[rank];
            int rank_number_data = destCountInnerNbr[rank];
            std::vector<int> rank_innerSize(rank_number_data);
            std::transform(global_data_inner_size.begin()+displsInnerNbr[rank],
                        global_data_inner_size.begin()+displsInnerNbr[rank]+destCountInnerNbr[rank],
                        rank_innerSize.begin(),[](int len){return len;});
            int displacement=0;
            for(int j=0;j<rank_innerSize.size();j++)
            {
                std::vector<DATA_Element> dat(rank_innerSize[j]);
                std::memcpy(dat.data(),&global_DATA_Elements[rank_DataElement_Start+displacement],
                            rank_innerSize[j]*sizeof(DATA_Element));
                displacement+=rank_innerSize[j];
                (*collectedData)[rank].push_back(transformElementaryToData(dat));
            }
            
            /*
            if(my_rank==0 && rank==2)
            {
                std::cout<<"After reorganize"<<std::endl;
                std::cout<<"Rank:"<<my_rank<<"  size:"<<rank_innerSize.size()<<std::endl;
                for(int i=0;i<rank_innerSize.size();i++)
                        std::cout<<" "<<rank_innerSize[i];
                std::cout<<"Rank:"<<my_rank<<"  size:"<<(*collectedData)[rank].size()<<std::endl;
                for(int i=0;i<(*collectedData)[rank].size();i++)
                        std::cout<<" "<<(*collectedData)[rank][i]; 
                std::cout<<std::endl;    
                fflush(stdout);
            }
            */
        }
    }

    return std::move(collectedData);
};
