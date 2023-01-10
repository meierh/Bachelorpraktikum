#include "GraphProperty.h"

std::unique_ptr<GraphProperty::AreaConnecMap> GraphProperty::areaConnectivityStrength
(
    const DistributedGraph& graph,
    unsigned int resultToRank
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    
    std::function<std::uint64_t(int,int)> date_get_function = std::bind(
        &DistributedGraph::get_node_area_localID, graph,std::placeholders::_1, std::placeholders::_2);
    auto collected_data = collectAlongEdges_InToOut<std::uint64_t>(
        graph,MPI_UINT64_T,date_get_function);
    
    std::unique_ptr<collectedData_ptr<std::uint64_t>> rank_to_databuffer = std::move(std::get<0>(collected_data));
    std::unique_ptr<collectedDataStructure_ptr> rank_to_NodeID_to_localInd = std::move(std::get<1>(collected_data));
    std::unique_ptr<collectedDataIndexes_ptr> treated_ranks_to_pair_ind_size_recv = std::move(std::get<2>(collected_data));
    
    for(auto iter = treated_ranks_to_pair_ind_size_recv->begin();iter!=treated_ranks_to_pair_ind_size_recv->end();iter++)
    {
        int rank = iter->first;
        int index = iter->second.first;
        int size = iter->second.second;
        auto min = std::min_element((*rank_to_databuffer)[index].begin(),(*rank_to_databuffer)[index].end());
        auto max = std::max_element((*rank_to_databuffer)[index].begin(),(*rank_to_databuffer)[index].end());

        //std::cout<<"rank:"<<rank<<"  min:"<<*min<<"  max:"<<*max<<std::endl;
    }
    
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
            
            int rank_local_ind = (*treated_ranks_to_pair_ind_size_recv)[otherRank].first;
            int nodeID_localInd = (*rank_to_NodeID_to_localInd)[rank_local_ind][node_local_ind];
            
            AreaLocalID target_area_ID(otherRank,(*rank_to_databuffer)[rank_local_ind][nodeID_localInd]);
            std::pair<AreaLocalID,AreaLocalID> area_to_area(source_area_ID,target_area_ID);

            // Non existend key has the value 0 because of value initialization
            // https://en.cppreference.com/w/cpp/container/unordered_map/operator_at
            areaIDConnecStrengthMapLocal[area_to_area]+=oEdge.weight;
        }
    }
    
    //Exchange of area local ind and strings
    //std::unordered_map<AreaLocalID,std::string,stdPair_hash> areaID_to_name;
    std::vector<int> area_names_char_dist;
    const std::vector<std::string> area_names = graph.get_local_area_names();
    for(const std::string& name : area_names)
    {
        area_names_char_dist.push_back(name.size());
    }
    std::vector<char> transmit_area_names;;
    for(const std::string& name : area_names)
    {
        for(int i=0;i<name.size();i++)
        {
            transmit_area_names.push_back(name[i]);
        }
    }
    

    /*
    std::cout<<my_rank<<"  ";
    for(int i=0;i<area_names.size();i++)
    {
        std::cout<<area_names_char_dist[i]<<"("<<area_names[i]<<")  ";
    }
    std::cout<<std::endl;
    */
    
    // exchange size of char index array
    int size = area_names_char_dist.size();
    std::vector<int> sizes;
    if(my_rank==resultToRank)
    {
        sizes.resize(MPIWrapper::get_number_ranks());
    }
    MPIWrapper::gather<int>(&size, sizes.data(), 1, MPI_INT, resultToRank);
    
    std::cout<<"Rank: "<<my_rank<<"[";
    for(int i=0;i<sizes.size();i++)
    {
        std::cout<<" "<<sizes[i];
    }
    std::cout<<"]"<<std::endl;
    
    MPIWrapper::barrier();

    //exchange index array
    std::vector<int> displsInt;
    std::vector<int> global_area_names_char_dist;
    if(my_rank==resultToRank)
    {
        displsInt.resize(MPIWrapper::get_number_ranks());
        int displacement = 0;
        for(int i=0;i<sizes.size();i++)
        {
            displsInt[i] = displacement;
            displacement+= sizes[i];
        }
        global_area_names_char_dist.resize(displacement);
    }
    MPIWrapper::gatherv<int>(area_names_char_dist.data(),size,
                            global_area_names_char_dist.data(),sizes.data(),displsInt.data(),
                            MPI_INT,resultToRank);
    std::vector<std::vector<int>> rank_to_area_names_char_dist;
    if(my_rank==resultToRank)
    {
        rank_to_area_names_char_dist.resize(MPIWrapper::get_number_ranks());
        for(int i=0;i<displsInt.size()-1;i++)
        {
            for(int j=displsInt[i];j<displsInt[i+1];j++)
            {
                rank_to_area_names_char_dist[i].push_back(global_area_names_char_dist[j]);
            }
        }
        for(int j=displsInt.back();j<global_area_names_char_dist.size();j++)
        {
            rank_to_area_names_char_dist.back().push_back(global_area_names_char_dist[j]);
        }
        
        for(int i=0;i<rank_to_area_names_char_dist.size();i++)
        {
            std::cout<<"From Rank: "<<i<<" [";
            for(int j=0;j<rank_to_area_names_char_dist[i].size();j++)
            {
                std::cout<<" "<<rank_to_area_names_char_dist[i][j];
            }
            std::cout<<"] "<<rank_to_area_names_char_dist[i].size()<<std::endl;
        }
        std::cout<<std::endl;
    }    
    
    fflush(stdout);
    MPIWrapper::barrier();
    
    std::vector<int> displsChar;
    std::vector<char> global_area_names_char;
    int sizeChar = transmit_area_names.size();
    std::vector<int> sizeCharArray;
    if(my_rank==resultToRank)
    {
        displsChar.resize(MPIWrapper::get_number_ranks());
        sizeCharArray.resize(MPIWrapper::get_number_ranks());
        int displacement = 0;
        for(int i=0;i<displsInt.size();i++)
        {
            displsChar[i] = displacement;
            sizeCharArray[i] = std::accumulate(rank_to_area_names_char_dist[i].begin(),
                                           rank_to_area_names_char_dist[i].end(),0);
            displacement+= sizeCharArray[i];
        }
        global_area_names_char.resize(displacement);
    }
    
    
    MPIWrapper::barrier();
    std::cout<<"Rank:"<<my_rank<<" - sizeChar"<<sizeChar<<std::endl;
    MPIWrapper::barrier();
    std::cout<<"Rank:"<<my_rank<<" - global_area_names_char.size()"<<global_area_names_char.size()<<std::endl<<std::endl;
    MPIWrapper::barrier();
    std::cout<<"Rank:"<<my_rank<<" - transmit_area_names: "<<transmit_area_names.size()<<" [";
    for(char c:transmit_area_names)
        std::cout<<c;
    std::cout<<"]"<<std::endl<<std::endl;
    MPIWrapper::barrier();
    std::cout<<"Rank:"<<my_rank<<" - sizeCharArray: "<<sizeCharArray.size()<<" [";
    for(int c:sizeCharArray)
        std::cout<<c<<" ";
    std::cout<<"]"<<std::endl;
    MPIWrapper::barrier();
    std::cout<<"Rank:"<<my_rank<<" - displsChar: "<<displsChar.size()<<" [";
    for(int c:displsChar)
        std::cout<<c<<" ";
    std::cout<<"]"<<std::endl;
    MPIWrapper::barrier();
    MPIWrapper::gatherv<char>(transmit_area_names.data(),sizeChar,
                              global_area_names_char.data(),sizeCharArray.data(),displsChar.data(),
                              MPI_CHAR,resultToRank);
    std::vector<std::vector<std::string>> rank_to_area_names;
    if(my_rank==resultToRank)
    {
        rank_to_area_names.resize(MPIWrapper::get_number_ranks());
        int displacement = 0;
        for(int i=0;i<rank_to_area_names_char_dist.size();i++)
        {
            for(int j=0;j<rank_to_area_names_char_dist[i].size();j++)
            {
                std::string name(&global_area_names_char[displacement],
                                 rank_to_area_names_char_dist[i][j]);
                rank_to_area_names[i].push_back(name);
                displacement+= rank_to_area_names_char_dist[i][j];
            }
        }
        
        for(int i=0;i<rank_to_area_names.size();i++)
        {
            std::cout<<"From Rank: "<<i<<" [";
            for(int j=0;j<rank_to_area_names[i].size();j++)
            {
                std::cout<<" "<<rank_to_area_names[i][j];
            }
            std::cout<<"] "<<rank_to_area_names[i].size()<<std::endl;
        }
        std::cout<<std::endl;
    }
    
    std::vector<std::pair<AreaLocalID,AreaLocalID>> area_to_area_list;
    std::vector<int> weightSum_list;
    for(auto keyValue = areaIDConnecStrengthMapLocal.begin();
        keyValue != areaIDConnecStrengthMapLocal.end();
        ++keyValue)
    {
        area_to_area_list.push_back(keyValue->first);
        weightSum_list.push_back(keyValue->second);
    }
    
    int area_connectivity_size = area_to_area_list.size();
    std::vector<int> area_connectivity_sizes;
    if(my_rank==resultToRank)
    {
        area_connectivity_sizes.resize(MPIWrapper::get_number_ranks());
    }
    MPIWrapper::gather<int>(&area_connectivity_size,area_connectivity_sizes.data(), 
                            1,MPI_INT,resultToRank);
    
    MPIWrapper::barrier();
    std::cout<<"Rank:"<<my_rank<<" - area_connectivity_sizes: "<<area_connectivity_sizes.size()<<" [";
    for(int c:area_connectivity_sizes)
        std::cout<<c<<" ";
    std::cout<<"]"<<std::endl;
    MPIWrapper::barrier();
    
    std::vector<std::pair<AreaLocalID,AreaLocalID>> global_area_to_area_list;
    std::vector<int> global_weightSum_list;
    std::vector<int> area_to_area_weights_displs;
    if(my_rank==resultToRank)
    {
        int global_area_connectivity_size = std::accumulate(area_connectivity_sizes.begin(),
                                                            area_connectivity_sizes.end(),0);
        global_area_to_area_list.resize(global_area_connectivity_size);
        global_weightSum_list.resize(global_area_connectivity_size);
        
        area_to_area_weights_displs.resize(MPIWrapper::get_number_ranks());
        int displacement = 0;
        for(int i=0;i<area_connectivity_sizes.size();i++)
        {
            area_to_area_weights_displs[i] = displacement;
            displacement+= area_connectivity_sizes[i];
        }
    }
    MPIWrapper::gatherv<int>(weightSum_list.data(),area_connectivity_size,
                            global_weightSum_list.data(),area_connectivity_sizes.data(),
                            area_to_area_weights_displs.data(),MPI_INT,resultToRank);
    
    MPIWrapper::barrier();
    std::cout<<"Rank:"<<my_rank<<" - global_area_to_area_list: "<<global_weightSum_list.size()<<" [";
    for(auto c :global_weightSum_list)
        std::cout<<" "<<c;
    std::cout<<"]"<<std::endl;
    MPIWrapper::barrier();

    MPIWrapper::gatherv<std::pair<AreaLocalID,AreaLocalID>>
                            (area_to_area_list.data(), area_connectivity_size,
                             global_area_to_area_list.data(),area_connectivity_sizes.data(),
                             area_to_area_weights_displs.data(),
                             MPIWrapper::MPI_stdPair_of_AreaLocalID,resultToRank);
       
    MPIWrapper::barrier();
    std::cout<<"Rank:"<<my_rank<<" - global_area_to_area_list: "<<global_area_to_area_list.size()<<" [";
    for(auto c :global_area_to_area_list)
        std::cout<<"{("<<c.first.first<<","<<c.first.second<<")-("<<c.second.first<<","<<c.second.second<<")}";
    std::cout<<"]"<<std::endl;
    MPIWrapper::barrier();

    fflush(stdout);
    MPIWrapper::barrier();
    
    areaIDConnecStrengthMapLocal.clear();
    for(int i=0;i<global_area_to_area_list.size();i++)
    {
        areaIDConnecStrengthMapLocal[global_area_to_area_list[i]]+=global_weightSum_list[i];
    }
    //Transfer data from ID to string
    auto result = std::make_unique<AreaConnecMap>();
  
    MPIWrapper::barrier();
    std::cout<<my_rank<<":  "<<areaIDConnecStrengthMapLocal.size()<<std::endl;
    MPIWrapper::barrier();
    fflush(stdout);
    //throw std::string("301");


    for(auto keyValue = areaIDConnecStrengthMapLocal.begin();
        keyValue != areaIDConnecStrengthMapLocal.end();
        ++keyValue)
    {
        if(my_rank!=resultToRank)
            throw std::string("Can not happen");
        
        AreaLocalID source_area_ID = keyValue->first.first;
        AreaLocalID target_area_ID = keyValue->first.second;
        int source_area_rank = source_area_ID.first;
        int source_area_id = source_area_ID.second;
        int target_area_rank = target_area_ID.first;
        int target_area_id = target_area_ID.second;
        
        assert(source_area_rank<rank_to_area_names.size() && source_area_rank>=0);
        assert(source_area_id<rank_to_area_names[source_area_rank].size() && source_area_id>=0);
        
        assert(target_area_rank<rank_to_area_names.size() && target_area_rank>=0);
        assert(target_area_id<rank_to_area_names[target_area_rank].size() && target_area_id>=0);
        
        std::string source_area_name = rank_to_area_names[source_area_rank][source_area_id];
        std::string target_area_name = rank_to_area_names[target_area_rank][target_area_id];
        auto area_to_area = std::pair<std::string,std::string>(source_area_name,target_area_name);
        (*result)[area_to_area]+=keyValue->second;
    }
    //throw std::string("330");

    return std::move(result);
}

std::unique_ptr<GraphProperty::AreaConnecMap> GraphProperty::areaConnectivityStrengthSingleProc
(
    const DistributedGraph& graph,
    unsigned int resultToRank
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    std::vector<uint64_t> node_numbers = MPIWrapper::all_gather(graph.get_number_local_nodes());

    ////////////////////////////////////////////// TAKEN FROM ABOVE //////////////////////////////////////////////
    //Gathering of area local ind and strings for the main process
    std::unordered_map<AreaLocalID,std::string,stdPair_hash> areaID_to_name;
    std::vector<int> area_names_char_dist;
    const std::vector<std::string> area_names = graph.get_local_area_names();
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
    ////////////////////////////////////////////// TAKEN FROM ABOVE //////////////////////////////////////////////

    auto result = std::make_unique<AreaConnecMap>();
    // Computation is performed by a single process:
    if(my_rank == 0)
    {
        const int& number_ranks = MPIWrapper::get_number_ranks();
        assert(node_numbers.size() == number_ranks);    // debug

        // Iterate over each rank...
        for(int r = 0; r < number_ranks; r++)
        {
            // ...and over each node from that rank
            for(int n = 0; n < node_numbers[r]; n++)
            {
                // Consider each outgoing edge and find out source and target areas
                const auto source_area_id = graph.get_node_area_localID(r, n);
                const std::vector<OutEdge>& oEdges = graph.get_out_edges(r, n);
                for(const OutEdge& oEdge : oEdges)
                {
                    std::uint64_t source_area_localID = graph.get_node_area_localID(r, n);
                    std::uint64_t target_area_localID = graph.get_node_area_localID(oEdge.target_rank, oEdge.target_id);
                    
                    AreaLocalID sourceArea = std::pair<uint64_t, uint64_t>(r, source_area_localID);
                    AreaLocalID targetArea = std::pair<uint64_t, uint64_t>(oEdge.target_rank, target_area_localID);
                    
                    std::string source_area_str = areaID_to_name[sourceArea];
                    std::string target_area_str = areaID_to_name[targetArea];

                    // Store all weights of the area pairs that realize a connection of two different areas 
                    // in the corresponding "area to area hash class" of the result map
                    if(source_area_str != target_area_str)
                    {
                        (*result)[{source_area_str, target_area_str}] += oEdge.weight;
                    }
                }   
            }
        }
    }
    return std::move(result);
}

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramm_constBinWidth
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
        return std::move(histogram);
    };
    return std::move(edgeLengthHistogramm(graph,bin_width_histogram_creator,resultToRank));
}

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramm_constBinCount
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
        double span_length = (max_length-min_length)*1.01; //achieve small overlap
        double bin_width = span_length/bin_count;
        double start_length = min_length-0.005*(max_length-min_length);

        auto histogram = std::make_unique<Histogram>(bin_count);
        for(int i=0;i<bin_count;i++)
        {
            (*histogram)[i].first.first = start_length;
            (*histogram)[i].first.second = start_length+bin_width;
            (*histogram)[i].second = 0;
            start_length = start_length+bin_width;
        }
        return std::move(histogram);
    };
    return std::move(edgeLengthHistogramm(graph,bin_count_histogram_creator,resultToRank));
}

std::vector<double> GraphProperty::networkTripleMotifs
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
    
    std::cout<<"Line 681 from process:"<<my_rank<<std::endl;

    std::vector<std::uint64_t> motifTypeCount(14,0);
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        std::unique_ptr<std::vector<threeMotifStructure>> this_node_motifs_results;
        this_node_motifs_results = threeMotifResults->getAnswersOfQuestionerNode(node_local_ind);
        
        for(int i=0;i<this_node_motifs_results->size();i++)
        {
            threeMotifStructure& one_motif = (*this_node_motifs_results)[i];
            for(int motifType=1;motifType<14;motifType++)
            {
                if(one_motif.isMotifTypeSet(motifType))
                    motifTypeCount[motifType]++;
            }
        }
    }
    
    std::vector<std::uint64_t> motifTypeCountTotal;
    if(my_rank==resultToRank)
    {
        motifTypeCountTotal.resize(14);
    }
    
    MPIWrapper::reduce<std::uint64_t>(motifTypeCount.data(),motifTypeCountTotal.data(),                                      
                                      14,MPI_UINT64_T,MPI_SUM,resultToRank);
    if(my_rank==resultToRank)
    {
        //Rotational invariant motifs where counted three times each
        assert(motifTypeCountTotal[7]%3==0);
        motifTypeCountTotal[7]/=3;
        assert(motifTypeCountTotal[13]%3==0);
        motifTypeCountTotal[13]/=3;        
    }
    std::vector<double> motifFraction;
    if(my_rank==resultToRank)
    {
        std::uint64_t total_number_of_motifs = std::accumulate(motifTypeCountTotal.begin(),motifTypeCountTotal.end(),0);
        motifFraction.resize(motifTypeCountTotal.size());
        motifFraction[0] = total_number_of_motifs;
        for(int motifType=1;motifType<14;motifType++)
        {
            motifFraction[motifType] = (double)motifTypeCountTotal[motifType] / (double)total_number_of_motifs;
        }
    }
    return motifFraction;
}

/*
double GraphProperty::computeModularity
(
    const DistributedGraph& graph
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const int number_ranks = MPIWrapper::get_number_ranks();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    const std::vector<std::string>& area_names = graph.get_local_area_names();

    // Find maximum name length
    int maxLengthName = std::max_element(area_names.begin(), area_names.end(),
                                               [](std::string a, std::string b) {return a.size() < b.size();}
                                               )->size();
    int global_maxLength_Name = MPIWrapper::all_reduce<int>(maxLengthName,MPI_INT,MPI_MAX);
    
    // Distribute number of names per rank to all ranks
    std::vector<int> global_number_of_names(number_ranks);
    int local_number_names = area_names.size();
    MPIWrapper::all_gather<int>(&local_number_names, global_number_of_names.data(), 1, MPI_INT);
    
    // Distribute length of names in all ranks to all ranks
    std::vector<int> global_sizes_of_names(std::accumulate(global_number_of_names.begin(),global_number_of_names.end(),0));
    std::vector<int> global_sizes_of_names_displ(number_ranks);
    std::partial_sum(global_number_of_names.begin(), global_number_of_names.end()-1,
                     global_sizes_of_names_displ.begin()+1, std::plus<int>());
    std::vector<int> local_sizes_of_names(area_names.size());
    std::transform(area_names.cbegin(),area_names.cend(),local_sizes_of_names.begin(),
                   [](std::string s){return s.size();});
    MPIWrapper::all_gatherv<int>(local_sizes_of_names.data(),local_number_names,
                                 global_sizes_of_names.data(),global_number_of_names.data(),
                                 global_sizes_of_names_displ.data(), MPI_INT);
    
    // Distribute characters of names in all ranks to all ranks
    std::vector<char> global_char_of_names(std::accumulate(global_sizes_of_names.begin(),global_sizes_of_names.end(),0));
    std::vector<int> global_char_of_names_displ(number_ranks);
    std::vector<int> global_sizes_of_char(number_ranks);
    int displacement = 0;
    for(int i=0;i<global_sizes_of_names_displ.size()-1;i++)
    {
        global_char_of_names_displ[i]=displacement;
        global_sizes_of_char[i]=std::accumulate(global_sizes_of_names.begin()+global_sizes_of_names_displ[i],
                                                global_sizes_of_names.begin()+global_sizes_of_names_displ[i+1],0);
        displacement+=global_sizes_of_char[i];
    }
    std::vector<char> local_char_of_names;
    std::for_each(area_names.cbegin(),area_names.cend(),
                  [&](std::string s){local_char_of_names.insert(local_char_of_names.cend(),s.cbegin(),s.cend());});
    MPIWrapper::all_gatherv<char>(local_char_of_names.data(),global_sizes_of_char[my_rank],
                                 global_char_of_names.data(),global_sizes_of_char.data(),
                                 global_char_of_names_displ.data(), MPI_CHAR);
    std::vector<std::vector<std::string>> area_names_list_of_ranks(number_ranks);
    for(int rank=0;rank<number_ranks;rank++)
    {
        int rank_char_begin = global_char_of_names_displ[rank];
        int rank_number_of_names = global_number_of_names[rank];
        std::vector<int> rank_sizes_of_names(rank_number_of_names);
        std::memcpy(rank_sizes_of_names.data(),&global_sizes_of_names[global_sizes_of_names_displ[rank]],
                    rank_number_of_names*sizeof(int));
        for(int wordNbr=0;wordNbr<rank_number_of_names;wordNbr++)
        {
            std::string name(&global_char_of_names[rank_char_begin],rank_sizes_of_names[wordNbr]);
            area_names_list_of_ranks[rank].push_back(name);
        }
    }
    
    //Create global name indices
    std::unordered_map<std::string,std::uint64_t> area_names_map;
    std::vector<std::string> area_names_list;
    for(std::vector<std::string>& rank_names : area_names_list_of_ranks)
    {
        for(std::string& name : rank_names)
        {
            auto status = area_names_map.insert(std::pair<std::string,std::uint64_t>(name,area_names_list.size()));
            if(status.second)
            {
                area_names_list.push_back(name);
            }
        }
    }
    
    //Create rank local area distance sum
    std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>>>
                 (const DistributedGraph& dg,std::uint64_t node_local_ind)>
        collect_adjacency_area_info = [&](const DistributedGraph& dg,std::uint64_t node_local_ind)
        {
            const int my_rank = MPIWrapper::get_my_rank();
            
            std::uint64_t node_area_localID = dg.get_node_area_localID(my_rank, node_local_ind);
            auto keyValue = area_names_map.find(area_names[node_area_localID]);
            assert(keyValue!=area_names_map.end()); 
            std::uint64_t area_global_ID = keyValue->second;
            
            auto outward_node_area = std::make_unique<std::vector<std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>>>();
            
            const std::vector<OutEdge>& oEdges = dg.get_out_edges(my_rank,node_local_ind);
            for(const OutEdge& oEdge : oEdges)
            {
                std::uint64_t rank = oEdge.target_rank;
                std::uint64_t id = oEdge.target_id;
                outward_node_area->push_back(std::tie<std::uint64_t,std::uint64_t,std::uint64_t>
                                                        (rank,id,area_global_ID));
            }
            return outward_node_area;
        };
        
    std::function<std::uint8_t(const DistributedGraph& dg,std::uint64_t node_local_ind,std::uint64_t area_global_ID)> 
        test_for_adjacent_equal_Area = 
                [&](const DistributedGraph& dg,std::uint64_t node_local_ind,std::uint64_t area_global_ID)
        {
            const int my_rank = MPIWrapper::get_my_rank();
            
            std::uint64_t node_area_localID = dg.get_node_area_localID(my_rank, node_local_ind);
            auto keyValue = area_names_map.find(area_names[node_area_localID]);
            assert(keyValue!=area_names_map.end());
            
            if(area_global_ID==keyValue->second)
            {
                return 1;
            }
            return 0;
        };
        
    std::unique_ptr<NodeToNodeQuestionStructure<std::uint64_t,std::uint8_t>> adjacency_results;
    adjacency_results = std::move(node_to_node_question<std::uint64_t,std::uint8_t>
                            (graph,MPI_UINT64_T,collect_adjacency_area_info,
                                   MPI_UINT8_T,test_for_adjacent_equal_Area));
    
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
    
    std::vector<std::vector<nodeModularityInfo>> areaGlobalID_to_node(area_names_list.size());
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {         
        std::uint64_t node_area_localID = graph.get_node_area_localID(my_rank, node_local_ind);
        auto keyValue = area_names_map.find(area_names[node_area_localID]);
        assert(keyValue!=area_names_map.end());
        std::uint64_t area_global_ID = keyValue->second;
    
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_local_ind);
        const std::vector<InEdge>& iEdges = graph.get_in_edges(my_rank,node_local_ind);
        nodeModularityInfo nodeInfo;
        nodeInfo.node_in_degree = iEdges.size();
        nodeInfo.node_out_degree = oEdges.size();
        nodeInfo.area_global_ID = area_global_ID;
        
        areaGlobalID_to_node[area_global_ID].push_back(nodeInfo);
    }
    std::vector<std::vector<nodeModularityInfo>> local_areaGlobalID_to_node;
    for(int area_global_ID=0;area_global_ID<area_names_list.size();area_global_ID++)
    {
        int calculationRank = area_global_ID%number_ranks;
        
        int area_local_size = areaGlobalID_to_node[area_global_ID].size();
        int total_size;
        MPIWrapper::reduce<int>(&total_size,&area_local_size,1,MPI_INT,MPI_SUM,calculationRank);
        
        if(my_rank==calculationRank)
        {
            local_areaGlobalID_to_node.push_back({});
            local_areaGlobalID_to_node.back().resize(total_size);
        }
        MPIWrapper::gather<nodeModularityInfo>(areaGlobalID_to_node[area_global_ID].data(),
                                               local_areaGlobalID_to_node.back().data(),area_local_size,
                                               MPIWrapper::MPI_nodeModularityInfo,calculationRank);
    }
    std::uint64_t local_in_out_degree_node_sum = 0;
    for(std::vector<nodeModularityInfo>& nodes_of_area : local_areaGlobalID_to_node)
    {
        for(int i=0;i<nodes_of_area.size();i++)
        {
            for(int j=0;j<nodes_of_area.size();j++)
            {
                if(i!=j)
                {
                    assert(nodes_of_area[i].area_global_ID==nodes_of_area[j].area_global_ID);
                    local_in_out_degree_node_sum+=(-(nodes_of_area[i].node_in_degree*nodes_of_area[j].node_out_degree));
                }
            }
        }
    }
    std::uint64_t global_in_out_degree_node_sum = MPIWrapper::all_reduce<std::uint64_t>(local_in_out_degree_node_sum,
                                                                                        MPI_UINT64_T,MPI_SUM);
    
    std::uint64_t local_m = number_local_nodes;
    std::uint64_t global_m = MPIWrapper::all_reduce<std::uint64_t>(local_m,MPI_UINT64_T,MPI_SUM);

    return (double)global_adjacency_sum/(double)(global_m*global_m) + (double)global_in_out_degree_node_sum/(double)(global_m);
}
*/

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramm
(
    const DistributedGraph& graph,
    std::function<std::unique_ptr<Histogram>(double,double)> histogram_creator,
    unsigned int resultToRank
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    
    std::function<Vec3d(int,int)> date_get_function = std::bind(
        &DistributedGraph::get_node_position, graph,std::placeholders::_1, std::placeholders::_2);
    auto collected_data = collectAlongEdges_InToOut<Vec3d>(graph,MPIWrapper::MPI_Vec3d,date_get_function);
    
    std::unique_ptr<collectedData_ptr<Vec3d>> rank_to_databuffer = std::move(std::get<0>(collected_data));
    std::unique_ptr<collectedDataStructure_ptr> rank_to_NodeID_to_localInd = std::move(std::get<1>(collected_data));
    std::unique_ptr<collectedDataIndexes_ptr> treated_ranks_to_pair_ind_size_recv = std::move(std::get<2>(collected_data));
    
    std::vector<double> edge_lengths;    
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        const Vec3d source_pos = graph.get_node_position(my_rank,node_local_ind);
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_local_ind);
        for(const OutEdge& oEdge : oEdges)
        {
            std::uint64_t otherRank = oEdge.target_rank;
            std::uint64_t otherID = oEdge.target_id;
            
            int rank_local_ind = (*treated_ranks_to_pair_ind_size_recv)[otherRank].first;
            int nodeID_localInd = (*rank_to_NodeID_to_localInd)[rank_local_ind][node_local_ind];
            
            Vec3d target_pos = (*rank_to_databuffer)[rank_local_ind][nodeID_localInd];
            edge_lengths.push_back((source_pos-target_pos).calculate_p_norm(2));
        }
    }
    
    const auto [min_length, max_length] = std::minmax_element(edge_lengths.begin(), edge_lengths.end());
    double global_min_length = *min_length;
    double global_max_length = *max_length;
    
    std::cout<<"edge_lengths.size: "<<my_rank<<": "<<edge_lengths.size()<<std::endl;
    std::cout<<"min_length "<<my_rank<<": "<<global_min_length<<std::endl;
    std::cout<<"max_length "<<my_rank<<": "<<global_max_length<<std::endl;
    MPIWrapper::barrier();
    
    global_min_length = MPIWrapper::all_reduce<double>(global_min_length,MPI_DOUBLE,MPI_MIN);
    global_max_length = MPIWrapper::all_reduce<double>(global_max_length,MPI_DOUBLE,MPI_MAX);
    
    MPIWrapper::barrier();
    std::cout<<"---------------------------------------------------------"<<std::endl;
    MPIWrapper::barrier();
    std::cout<<"edge_lengths.size: "<<my_rank<<": "<<edge_lengths.size()<<std::endl;
    std::cout<<"min_length "<<my_rank<<": "<<global_min_length<<std::endl;
    std::cout<<"max_length "<<my_rank<<": "<<global_max_length<<std::endl;
    MPIWrapper::barrier();
    
    std::unique_ptr<Histogram> histogram = histogram_creator(global_min_length,global_max_length);
    std::pair<double,double> span = histogram->front().first;
    double bin_width = span.second - span.first;
    
    std::cout<<"histogram->size() "<<my_rank<<": "<<histogram->size()<<std::endl;
    std::cout<<"bin_width "<<my_rank<<": "<<bin_width<<std::endl;
    std::cout<<"min_length "<<my_rank<<": "<<global_min_length<<std::endl;
    std::cout<<"max_length "<<my_rank<<": "<<global_max_length<<std::endl;
    MPIWrapper::barrier();
    std::cout<<"histogram->front() "<<my_rank<<": "<<"("<<histogram->front().first.first<<","<<histogram->front().first.second<<")"<<std::endl;
    std::cout<<"histogram->back() "<<my_rank<<": "<<"("<<histogram->back().first.first<<","<<histogram->back().first.second<<")"<<std::endl;

    for(const double length : edge_lengths)
    {
        int index = (length-global_min_length)/bin_width;
        assert(index<histogram->size());
        (*histogram)[index].second++;
    }
    
    MPIWrapper::barrier();
    //throw std::string("572");
    
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
    
    MPIWrapper::barrier();
    //throw std::string("578");
    
    MPIWrapper::reduce<std::uint64_t>(histogram_pure_count_src.data(),histogram_pure_count_dest.data(),
                                          histogram->size(),MPI_UINT64_T,MPI_SUM,resultToRank);
    MPIWrapper::barrier();
    MPIWrapper::barrier();
    std::cout<<"Rank:"<<my_rank<<" - histogram_pure_count_dest: "<<histogram_pure_count_dest.size()<<" [";
    for(auto c :histogram_pure_count_dest)
        std::cout<<c<<" ";
    std::cout<<"]"<<std::endl;
    MPIWrapper::barrier();
    //throw std::string("589");
    if(my_rank==resultToRank)
    {
        for(int i=0;i<histogram->size();i++)
        {
            (*histogram)[i].second = histogram_pure_count_dest[i];
        }
    }
    else
    {
        histogram.reset(nullptr);
    }
    //throw std::string("609");

    return std::move(histogram);
}

template<typename DATA> 
std::tuple<
std::unique_ptr<GraphProperty::collectedData_ptr<DATA>>,
std::unique_ptr<GraphProperty::collectedDataStructure_ptr>,
std::unique_ptr<GraphProperty::collectedDataIndexes_ptr>
>
GraphProperty::collectAlongEdges_InToOut
(
    const DistributedGraph& graph,
    MPI_Datatype datatype,
    std::function<DATA(int,int)> date_get_function
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const int number_ranks = MPIWrapper::get_number_ranks();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    int ownRankRecvInd;
    
/* prepare buffer for data to receive from other ranks
 */
    /* map: rank to receive from -> (index of rank to receive from in "ranks_recv",
     *                               number of outEdges ending in the rank to receive from)
     */ 
    auto treated_ranks_to_pair_ind_size_recv = std::make_unique<collectedDataIndexes_ptr>();
    // list of ranks to receive from
    std::vector<int> ranks_recv;
    /* list analog to ranks_recv - map: node_local_ind -> respective index in  
     * rank_ind_to_data_ind_list_recv
     */
    auto rank_ind_NodeID_to_indexInData = std::make_unique<collectedDataStructure_ptr>();
    
    
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_local_ind);
        for(const OutEdge& oEdge : oEdges)
        {
            const auto rank_to_pair_rankInd_nbrOutEdges = treated_ranks_to_pair_ind_size_recv->find(oEdge.target_rank);
            
            if(rank_to_pair_rankInd_nbrOutEdges != treated_ranks_to_pair_ind_size_recv->end())
            // If rank was already encoutered due to other outEdge 
            {
                (*rank_ind_NodeID_to_indexInData)[rank_to_pair_rankInd_nbrOutEdges->second.first].insert
                (
                    std::pair<std::uint64_t,int>
                    (
                        node_local_ind,
                        (*treated_ranks_to_pair_ind_size_recv)[oEdge.target_rank].second
                    )
                );
            }
            else
            {
                treated_ranks_to_pair_ind_size_recv->insert
                (
                    std::pair<int,std::pair<int,int>>(oEdge.target_rank,{ranks_recv.size(),0})
                );
                ranks_recv.push_back(oEdge.target_rank);
                rank_ind_NodeID_to_indexInData->push_back({{node_local_ind,0}});
            }
            (*treated_ranks_to_pair_ind_size_recv)[oEdge.target_rank].second++;
        }
    }
    
// Testing Routine Start
    for(int i=0;i<rank_ind_NodeID_to_indexInData->size();i++)
        for(auto iter = (*rank_ind_NodeID_to_indexInData)[i].begin();iter!=(*rank_ind_NodeID_to_indexInData)[i].end();iter++)
        {
            int node_local_ind = iter->first;
            int index_in_Data = iter->second;
            auto a = treated_ranks_to_pair_ind_size_recv->find(ranks_recv[i]);
            if(index_in_Data>=0 && a!=treated_ranks_to_pair_ind_size_recv->end() && index_in_Data<a->second.second){}
            else
                throw 4;
        }
    for(int i=0;i<ranks_recv.size();i++)
        if(auto triple = treated_ranks_to_pair_ind_size_recv->find(ranks_recv[i]); triple!=treated_ranks_to_pair_ind_size_recv->end())
        {
            if(triple->second.first != i)
                throw 3;
        }
    std::unordered_set<int> unique;
    for(int a: ranks_recv)
        if(unique.find(a)==unique.end())
            unique.insert(a);
        else
        {
            for(int a: ranks_recv)
                std::cout << a << ' ';
            throw 2;
        }
// Testing Routine End

    
    // list analog to recv ranks: list of receive buffer
    auto rank_ind_to_data_ind_list_recv = std::make_unique<collectedData_ptr<DATA>>(ranks_recv.size());
    for(int i=0; i<ranks_recv.size(); i++)
    {
        int rank = ranks_recv[i];
        assert(treated_ranks_to_pair_ind_size_recv->find(rank)!=treated_ranks_to_pair_ind_size_recv->end());
        (*rank_ind_to_data_ind_list_recv)[i].resize((*treated_ranks_to_pair_ind_size_recv)[rank].second);
    }

/* setup nonblocking recv for areaID data from other ranks
 */
    std::vector<MPI_Request> requestRecv;
    for(int i=0; i<ranks_recv.size(); i++)
    {
        int source_rank = ranks_recv[i];
        assert(treated_ranks_to_pair_ind_size_recv->find(source_rank)!=treated_ranks_to_pair_ind_size_recv->end());
        int ind = (*treated_ranks_to_pair_ind_size_recv)[source_rank].first;
        assert(ind<rank_ind_to_data_ind_list_recv->size() && ind>=0);
        if(source_rank == my_rank)
        {
            ownRankRecvInd = ind;
            continue;
        }
        int count = (*treated_ranks_to_pair_ind_size_recv)[source_rank].second;
        DATA* buffer = (*rank_ind_to_data_ind_list_recv)[ind].data();
        int tag = cantorPair(source_rank,my_rank);
        assert(count == (*rank_ind_to_data_ind_list_recv)[ind].size());
        requestRecv.emplace_back();
        MPIWrapper::Irecv(buffer,count,datatype,source_rank,tag,&requestRecv.back());
    }

    
/* prepare areaID data to send to other ranks
 */
    // maps: rank ID -> respective index in ranks_send
    std::unordered_map<int,int> treated_ranks_send;
    // list analog to ranks_send: list of send buffers
    std::vector<std::vector<DATA>> rank_ind_to_data_list_send;
    // list of ranks to send to
    std::vector<int> ranks_send;
    
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        const std::vector<InEdge>& iEdges = graph.get_in_edges(my_rank,node_local_ind);
        const DATA date = date_get_function(my_rank,node_local_ind);
        for(const InEdge& iEdge : iEdges)
        {
            const auto rank_ind = treated_ranks_send.find(iEdge.source_rank);
            if(rank_ind != treated_ranks_send.end())
            {
                assert(rank_ind->second<rank_ind_to_data_list_send.size() && rank_ind->second>=0);
                rank_ind_to_data_list_send[rank_ind->second].push_back(date);
            }
            else
            {     
                treated_ranks_send.insert(std::pair<int,int>(iEdge.source_rank,ranks_send.size()));
                rank_ind_to_data_list_send.push_back({date});
                ranks_send.push_back(iEdge.source_rank);
            }
        }
    }
    
// Testing Routine Start
    for(auto iter=treated_ranks_send.begin();iter!=treated_ranks_send.end();iter++)
        if(ranks_send[iter->second] != iter->first)
            throw 5;
    
    unique.clear();
    for(int a: ranks_send)
        if(unique.find(a)==unique.end())
            unique.insert(a);
        else
        {
            for(int a: ranks_send)
                std::cout << a << ' ';
            throw 6;
        }
// Testing Routine End

    MPIWrapper::barrier();

/* setup nonblocking send for areaID data to other ranks
 */
    std::vector<MPI_Request> requestSend;
    for(int i=0; i<ranks_send.size(); i++)
    {
        int target_rank = ranks_send[i];
        int ind = treated_ranks_send[target_rank];
        assert(ind<rank_ind_to_data_list_send.size() && ind>=0);
        if(target_rank == my_rank)
        {
            (*rank_ind_to_data_ind_list_recv)[ownRankRecvInd] = rank_ind_to_data_list_send[ind];
            continue;
        }               
        int count = rank_ind_to_data_list_send[ind].size();
        DATA* buffer = rank_ind_to_data_list_send[ind].data();
        int tag = cantorPair(my_rank,target_rank);
        requestSend.emplace_back();
        MPIWrapper::Isend(buffer,count,datatype,target_rank,tag,&requestSend.back());
    }
 
/* Wait for send and recv completion
 */
    MPIWrapper::Waitall(requestRecv.size(),requestRecv.data());
    MPIWrapper::Waitall(requestSend.size(),requestSend.data());

    /*
    for(int i=0; i<rank_ind_to_data_list_send.size();i++)
    {
        int rank = ranks_send[i];
        auto min = std::min_element(rank_ind_to_data_list_send[i].begin(),rank_ind_to_data_list_send[i].end());
        auto max = std::max_element(rank_ind_to_data_list_send[i].begin(),rank_ind_to_data_list_send[i].end());

        std::cout<<"send rank:"<<rank<<"  min:"<<*min<<"  max:"<<*max<<std::endl;
    }
    
    for(auto iter = treated_ranks_to_pair_ind_size_recv->begin();iter!=treated_ranks_to_pair_ind_size_recv->end();iter++)
    {
        int rank = iter->first;
        int index = iter->second.first;
        int size = iter->second.second;
        auto min = std::min_element((*rank_ind_to_data_ind_list_recv)[index].begin(),(*rank_ind_to_data_ind_list_recv)[index].end());
        auto max = std::max_element((*rank_ind_to_data_ind_list_recv)[index].begin(),(*rank_ind_to_data_ind_list_recv)[index].end());

        std::cout<<"recv rank:"<<rank<<"  min:"<<*min<<"  max:"<<*max<<std::endl;
    }
    //std::cout<<"Rank:"<<my_rank<<" Number of nodes is:" << number_local_nodes << '\n';
    */
    
    return std::make_tuple
    (
        std::move(rank_ind_to_data_ind_list_recv),
        std::move(rank_ind_NodeID_to_indexInData),
        std::move(treated_ranks_to_pair_ind_size_recv)
    );
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
    const int my_rank = MPIWrapper::get_my_rank();
    const int number_ranks = MPIWrapper::get_number_ranks();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    
    //std::cout<<"Line 1094 from process:"<<my_rank<<std::endl;
    
    // Create Questioners structure
    auto questioner_structure = std::make_unique<GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter>>();
    for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
    {
        questioner_structure->addQuestionsFromOneNodeToSend(generateAddressees(graph,node_local_ind),node_local_ind);
    }
    questioner_structure->finalizeAddingQuestionsToSend();
    
    for(int index=0; index<questioner_structure->list_index_to_adressee_rank.size(); index++)
    {
        int targetRank = questioner_structure->list_index_to_adressee_rank[index];
        for(int j=0;j<questioner_structure->nodes_to_ask_question[index].size();j++)
        {
            int targetNode = questioner_structure->nodes_to_ask_question[index][j];
            int sourceNode = questioner_structure->nodes_that_ask_the_question[index][j];
            threeMotifStructure struc = questioner_structure->question_parameters[index][j];
            assert(struc.node_1_rank==my_rank);
            assert(struc.node_1_local==sourceNode);
            assert(struc.node_2_rank==targetRank);
            assert(struc.node_2_local==targetNode);
        }
    }
    
    //std::cout<<"Line 1103 from process:"<<my_rank<<std::endl;
    
    // Distribute number of questions to each rank
    std::vector<int>& send_ranks_to_nbrOfQuestions = questioner_structure->get_adressee_ranks_to_nbrOfQuestions();
    
    /*
    MPIWrapper::barrier();
    //std::cout<<"send_ranks_to_nbrOfQuestions  Rank:"<<my_rank<<"  ";
    for(int nbrToRank: send_ranks_to_nbrOfQuestions)
    {
        std::cout<<nbrToRank<<"  ";
    }
    std::cout<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    */
    
    std::vector<int> global_ranks_to_nbrOfQuestions(number_ranks*number_ranks);    
    std::vector<int> destCounts_ranks_to_nbrOfQuestions(number_ranks,number_ranks);
    std::vector<int> displ_ranks_to_nbrOfQuestions(number_ranks);
    for(int index = 0;index<displ_ranks_to_nbrOfQuestions.size();index++)
    {
        displ_ranks_to_nbrOfQuestions[index]=number_ranks*index;
    }
    MPIWrapper::all_gatherv<int>(send_ranks_to_nbrOfQuestions.data(), number_ranks,
                                global_ranks_to_nbrOfQuestions.data(), destCounts_ranks_to_nbrOfQuestions.data(), displ_ranks_to_nbrOfQuestions.data(), MPI_INT);

    /*
    MPIWrapper::barrier();
    std::cout<<"Line 1117 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    */
    
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
        
        for(int j=0;j<nodes_to_ask_question_for_rank.size();j++)
        {
            int targetNode = nodes_to_ask_question_for_rank[j];
            threeMotifStructure struc = question_parameters_for_rank[j];
            assert(struc.node_1_rank==my_rank);
            assert(struc.node_2_rank==rank);
            assert(struc.node_2_local==targetNode);
        }
        
        
        /*
        MPIWrapper::barrier();
        std::cout<<"Rank:"<<my_rank<<" - "<<count<<std::endl;
        fflush(stdout);
        MPIWrapper::barrier();
        if(my_rank==rank)
        {
            std::cout<<"Rank:"<<my_rank<<" -- ";
            for(int nbrToRank: recv_ranks_to_nbrOfQuestions)
            {
                std::cout<<nbrToRank<<"  ";
            }
            std::cout<<std::endl;
        }
        fflush(stdout);
        MPIWrapper::barrier();
        */
        
        MPIWrapper::gatherv<std::uint64_t>(nodes_to_ask_question_for_rank.data(), count,
                                           my_rank_total_nodes_to_ask_question.data(),
                                           recv_ranks_to_nbrOfQuestions.data(), displ_recv_ranks_to_nbrOfQuestions.data(),MPI_UINT64_T,rank);
        
        MPIWrapper::gatherv<Q_parameter>(question_parameters_for_rank.data(), count,
                                         my_rank_total_question_parameters.data(),
                                         recv_ranks_to_nbrOfQuestions.data(), displ_recv_ranks_to_nbrOfQuestions.data(),MPI_Q_parameter,rank);
    }
    
    MPIWrapper::barrier();
    std::cout<<"Line 1423 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    //throw std::string("1419");

    // Create Adressees structure
    GraphProperty::NodeToNodeQuestionStructure<Q_parameter,A_parameter> adressee_structure;
    adressee_structure.setQuestionsReceived(my_rank_total_nodes_to_ask_question,my_rank_total_question_parameters,
                                            recv_ranks_to_nbrOfQuestions,displ_recv_ranks_to_nbrOfQuestions);
    
    MPIWrapper::barrier();
    std::cout<<"Line 1434 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    //throw std::string("1437");
    
    // Compute the answers
    adressee_structure.computeAnswersToQuestions(graph,generateAnswers);
    
    MPIWrapper::barrier();
    std::cout<<"Line 1443 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    
    // Distribute answers to each rank
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
    
    MPIWrapper::barrier();
    std::cout<<"Line 1473 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    throw std::string("1476");
    
    questioner_structure->setAnswers(my_rank_total_answer_parameters,send_ranks_to_nbrOfAnswers,
                                    displ_send_ranks_to_nbrOfAnswers);

    MPIWrapper::barrier();
    std::cout<<"Line 1482 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    throw std::string("1485");
    
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
            assert(outerIndex<nodes_to_ask_question.size() && outerIndex>=0);
            nodes_to_ask_question[outerIndex].push_back(target_local_node);
            assert(outerIndex<nodes_that_ask_the_question.size() && outerIndex>=0);
            nodes_that_ask_the_question[outerIndex].push_back(questioner);
            assert(outerIndex<question_parameters.size() && outerIndex>=0);
            question_parameters[outerIndex].push_back(Q_parameter_struct);
            
            std::uint64_t innerIndex = nodes_to_ask_question.size();
            
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
    for(int rank=0;rank<rank_size.size();rank++)
    {
        int nbr_of_answers = rank_size[rank];
        if(nbr_of_answers > 0)
        {
            auto keyValue = rank_to_outerIndex.find(rank);
            assert(keyValue!=rank_to_outerIndex.end());
            int outerIndex = keyValue->second;
            
            answers_to_questions.push_back({});
            answers_to_questions.back().resize(nbr_of_answers);
            std::memcpy(&answers_to_questions[outerIndex],&total_answers[rank_displ[rank]],nbr_of_answers);
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
        std::cout<<"Rank:"<<MPIWrapper::get_my_rank()<<"  "<<i<<"/"<<question_parameters.size()<<"  Generated answer for:"<<answers_to_questions[i].size()<<std::endl; 
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
        assert(outerIndex<nodes_that_ask_the_question.size());
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
        keyValue->first == node_local_ind;
        keyValue++)
    {
        std::uint64_t outerIndex = keyValue->second.first;
        std::uint64_t innerIndex = keyValue->second.second;
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
    
    
    // Testing
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
}

