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
    std::unordered_map<AreaLocalID,std::string,stdPair_hash> areaID_to_name;
    std::vector<int> area_names_char_dist;
    const std::vector<std::string> area_names = graph.get_local_area_names();
    for(const std::string& name : area_names)
    {
        area_names_char_dist.push_back(name.size());
    }
    int area_names_totalLength = std::accumulate(area_names_char_dist.begin(),area_names_char_dist.end(),0);
    auto transmit_area_names = std::make_unique<char[]>(area_names_totalLength);

    /*
    std::cout<<my_rank<<"  ";
    for(int i=0;i<area_names.size();i++)
    {
        std::cout<<area_names_char_dist[i]<<"("<<area_names[i]<<")  ";
    }
    std::cout<<std::endl;
    */
    
    if(my_rank==resultToRank)
    {
        for(int otherRank=0; otherRank<MPIWrapper::get_number_ranks(); otherRank++)
        {
            if(otherRank==resultToRank)
                continue;
            
            int nbr_of_area_names;
            int tag1 = cantorPair(1,cantorPair(otherRank,my_rank));
            std::cout<<my_rank<<" try to get nbr_of_area_names from "<<otherRank<<" tag1:"<<tag1<<std::endl;
            MPIWrapper::Recv(&nbr_of_area_names,1,MPI_INT,otherRank,tag1);
            
            std::cout<<my_rank<<" recv "<<nbr_of_area_names<<" from "<<otherRank<<" tag1:"<<tag1<<std::endl;
            std::vector<int> area_names_char_dist_other(nbr_of_area_names);
            area_names_char_dist_other.resize(nbr_of_area_names);
            std::cout<<my_rank<<" recv: "<<area_names_char_dist_other.size()<<std::endl;
            int tag2 = cantorPair(2,cantorPair(otherRank,my_rank));
            
            int buffer[nbr_of_area_names*2];
            MPIWrapper::Recv(&buffer,nbr_of_area_names,MPI_INT,otherRank,tag2);
            
            std::cout<<my_rank<<" -recv: "<<area_names_char_dist_other.size()<<"  tag2:"<<tag2<<std::endl;
            for(int dist:area_names_char_dist_other)
                std::cout<<dist<<" ";
            std::cout<<std::endl;
            int area_names_totalLength_other = std::accumulate(area_names_char_dist.begin(),area_names_char_dist.end(),0);
            std::cout<<my_rank<<" totalLength: "<<area_names_totalLength_other<<std::endl;
            auto transmit_area_names_other = std::make_unique<char[]>(area_names_totalLength_other);
            int tag3 = cantorPair(3,cantorPair(otherRank,my_rank));
            MPIWrapper::Recv(transmit_area_names_other.get(),area_names_totalLength_other,MPI_CHAR,otherRank,tag3);
            
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
        int tag1 = cantorPair(1,cantorPair(my_rank,resultToRank));
        MPIWrapper::Send(&size,1,MPI_INT,resultToRank,tag1);

        std::cout<<my_rank<<" send size:"<<size<<" "<<area_names_char_dist.size()<<" tag1:"<<tag1<<std::endl;
        int tag2 = cantorPair(2,cantorPair(my_rank,resultToRank));
        MPIWrapper::Send(area_names_char_dist.data(),size,MPI_INT,resultToRank,tag2);
        std::cout<<my_rank<<" send int array"<<" "<<area_names_char_dist.size()<<" tag2:"<<tag2<<std::endl;

        int tag3 = cantorPair(3,cantorPair(my_rank,resultToRank));
        MPIWrapper::Send(transmit_area_names.get(),area_names_totalLength,MPI_CHAR,resultToRank,tag3);
        std::cout<<my_rank<<" send char array"<<" tag3:"<<tag3<<std::endl;

    }
    
    fflush(stdout);
    MPIWrapper::barrier();
    throw 117;
    
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

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramm
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

std::unique_ptr<GraphProperty::Histogram> GraphProperty::edgeLengthHistogramm
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
        return std::move(histogram);
    };
    return std::move(edgeLengthHistogramm(graph,bin_count_histogram_creator,resultToRank));
}

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
    global_min_length = MPIWrapper::all_reduce<double>(global_min_length,MPI_DOUBLE,MPI_MIN);
    global_max_length = MPIWrapper::all_reduce<double>(global_max_length,MPI_DOUBLE,MPI_MAX);
    
    std::unique_ptr<Histogram> histogram = histogram_creator(global_min_length,global_max_length);
    std::pair<double,double> span = histogram->front().first;
    double bin_width = span.second - span.first;

    for(const double length : edge_lengths)
    {
        int index = (length-global_min_length)/bin_width;
        (*histogram)[index].second++;
    }
    
    std::vector<std::uint64_t> histogram_pure_count_src(histogram->size());
    for(int i=0;i<histogram->size();i++)
    {
        histogram_pure_count_src[i] = (*histogram)[i].second;
    }
    std::vector<std::uint64_t> histogram_pure_count_dest;
    if(my_rank==resultToRank)
        histogram_pure_count_dest.resize(histogram->size());
    
    MPIWrapper::reduce<std::uint64_t>(histogram_pure_count_src.data(),histogram_pure_count_dest.data(),
                                          histogram->size(),MPI_UINT64_T,MPI_SUM,resultToRank);
    
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

