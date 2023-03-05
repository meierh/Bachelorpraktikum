#include "NetworkMotifs.h"

std::vector<long double> NetworkMotifs::compute_network_TripleMotifs
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
                            StdPair_hash> adjacent_nodes;
            
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
                            StdPair_hash> adjacent_nodes;
            
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
    
    std::unique_ptr<CommunicationPatterns::NodeToNodeQuestionStructure<threeMotifStructure,threeMotifStructure>> threeMotifResults;
    threeMotifResults = CommunicationPatterns::node_to_node_question<threeMotifStructure,threeMotifStructure>
                            (graph,MPIWrapper::MPI_threeMotifStructure,collect_possible_networkMotifs_oneNode,
                                   MPIWrapper::MPI_threeMotifStructure,evaluate_correct_networkMotifs_oneNode);
    
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
        if(motifTypeCount[7]%3 != 0) std::cout << "error: motifTypeCount[7]%3 != 0 ==> " << motifTypeCount[7] << std::endl; 
        motifTypeCountTotal[7]/=3;
        if(motifTypeCount[13]%3 != 0) std::cout << "error: motifTypeCount[13]%3 != 0 ==> " << motifTypeCount[13] << std::endl; 
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

        // Print NetworkMotifs:
        for (size_t i = 0; i < motifFraction.size(); i++) {
            std::cout << "motifFraction[" << i << "] = " << motifFraction[i] << std::endl;
        }
    }
    MPIWrapper::barrier();
    std::cout<<"Line 751 from process:"<<my_rank<<std::endl;
    fflush(stdout);
    MPIWrapper::barrier();
    //throw std::string("751");
    return motifFraction;
}



std::vector<long double> NetworkMotifs::compute_network_TripleMotifs_SingleProc
(
    const DistributedGraph& graph,
    unsigned int result_rank
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    int number_ranks = MPIWrapper::get_number_ranks();
    std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    auto total_number_nodes = NodeCounter::all_count_nodes(graph);
    
    // Main rank gathers other ranks number of nodes
    std::vector<std::uint64_t> number_nodes_of_ranks(number_ranks);
    MPIWrapper::gather<uint64_t>(&number_local_nodes, number_nodes_of_ranks.data(), 1, MPI_UINT64_T, result_rank);
    
    //Print all out edges AND check if there are nodes with self-referencing edges
    /*if(my_rank == result_rank) {
        
        std::unordered_map<std::tuple<std::pair<std::uint64_t, std::uint64_t>, std::pair<std::uint64_t, std::uint64_t>>, int, StdPair_hash> edges;
        for(int r = 0; r < number_ranks; r++) {
            for(std::uint64_t n = 0; n < number_nodes_of_ranks[r]; n++) {
                auto out_edges = graph.get_out_edges(r, n);
                for(auto out_edge : out_edges){
                    //std::cout << "edge: " << r << "," << n << " -> " << out_edge.target_rank << "," << out_edge.target_id << std::endl;
                    if(out_edge.target_rank == r && out_edge.target_id == n) {
                        std::cout << "error: node with self referencing out_edge found: node(" << r << ", " << n << ")" << std::endl;
                    }
                }
                auto in_edges = graph.get_in_edges(r, n);
                for(auto in_edge : in_edges){
                    if(in_edge.source_rank == r && in_edge.source_id == n) {
                        std::cout << "error: node with self referencing in_edge found: node(" << r << ", " << n << ")" << std::endl;
                    }
                }
            }
        }
        std::cout << std::endl;  
    }*/

    // Prepare structure
    std::vector<std::uint64_t> motifTypeCount(14, 0);

    if(my_rank == result_rank) {

        for(int current_rank = 0; current_rank < number_ranks; current_rank++) {
            std::cout << "Scanning nodes of rank " << current_rank+1 << " of " << number_ranks << " ..." << std::endl;
            for(std::uint64_t current_node = 0; current_node < number_nodes_of_ranks[current_rank]; current_node++) {
                
                //std::cout << "next node: " << current_rank << "," << current_node << std::endl;

                const std::vector<OutEdge>& oEdges = graph.get_out_edges(current_rank, current_node);
                const std::vector<InEdge>& iEdges = graph.get_in_edges(current_rank, current_node);
                
                std::unordered_map<std::pair<std::uint64_t, std::uint64_t>,
                                std::pair<bool,bool>,
                                StdPair_hash> adjacent_nodes;
                
                for(const OutEdge& oEdge : oEdges) {
                    std::pair<std::uint64_t, std::uint64_t> node_key(oEdge.target_rank, oEdge.target_id);
                    std::pair<bool,bool>& value = adjacent_nodes[node_key];
                    assert(!value.first);   
                    value.first = true;
                }
                
                for(const InEdge& iEdge : iEdges) {
                    std::pair<std::uint64_t,std::uint64_t> node_key(iEdge.source_rank,iEdge.source_id);
                    std::pair<bool,bool>& value = adjacent_nodes[node_key];
                    assert(!value.second);
                    value.second = true;
                }

                //Print adjacent_nodes
                /*std::unordered_map<std::pair<std::uint64_t, std::uint64_t>,  // [rank, node] --> [out, in]
                                std::pair<bool,bool>,
                                StdPair_hash>::iterator it;
                for (it = adjacent_nodes.begin(); it != adjacent_nodes.end(); it++) {
                    std::cout << "(" <<it->first.first << "," << it->first.second 
                              << ") --> " << it->second.first << " " << it->second.second << std::endl;
                }*/

                for(auto iterOuter = adjacent_nodes.begin(); iterOuter != adjacent_nodes.end(); iterOuter++) {
                    
                    // Prrevent overlapping of outer and inner 
                    // -> consider only each tripple-node-set (ignore permutations)
                    std::pair<std::uint64_t,std::uint64_t> node_Outer_key = iterOuter->first;
                    auto iterInner = iterOuter;
                    iterInner++;
                    for(iterInner; iterInner != adjacent_nodes.end(); iterInner++) {
                        
                        std::pair<std::uint64_t,std::uint64_t> node_Inner_key = iterInner->first;
                        if(node_Inner_key != node_Outer_key) {
                            // Exclude nodes with self-referencing edges
                            if((current_rank == node_Outer_key.first && current_node == node_Outer_key.second) ||
                             (current_rank == node_Inner_key.first && current_node == node_Inner_key.second) ||
                             (node_Outer_key.first == node_Inner_key.first && node_Outer_key.second == node_Inner_key.second)){
                                continue;
                            }

                            threeMotifStructure motifStruc;
                            motifStruc.node_1_rank = current_rank;
                            motifStruc.node_1_local = current_node;
                            motifStruc.node_2_rank = node_Outer_key.first;
                            motifStruc.node_2_local = node_Outer_key.second;
                            motifStruc.node_3_rank = node_Inner_key.first;
                            motifStruc.node_3_local = node_Inner_key.second;
                            
                            bool exists_edge_node1_to_node2 = iterOuter->second.first;
                            bool exists_edge_node2_to_node1 = iterOuter->second.second;
                            bool exists_edge_node1_to_node3 = iterInner->second.first;
                            bool exists_edge_node3_to_node1 = iterInner->second.second;

                            std::uint16_t exists_edge_bitArray = 0;
                            exists_edge_bitArray |= exists_edge_node1_to_node2 ? 1 : 0;
                            exists_edge_bitArray |= exists_edge_node2_to_node1 ? 2 : 0;
                            exists_edge_bitArray |= exists_edge_node1_to_node3 ? 4 : 0;
                            exists_edge_bitArray |= exists_edge_node3_to_node1 ? 8 : 0;

                            std::uint16_t exists_edge_bitArray_updated = update_edge_bitArray(graph, exists_edge_bitArray, 
                                        motifStruc.node_2_rank, motifStruc.node_2_local, motifStruc.node_3_rank, motifStruc.node_3_local);;

                            //std::cout << "node1=" << motifStruc.node_1_rank << "." << motifStruc.node_1_local << ", node2=" << motifStruc.node_2_rank << "." << motifStruc.node_2_local << ", node3=" << motifStruc.node_3_rank << "." << motifStruc.node_3_local << std::endl;
                            //std::cout << "exists_edge_bitArray = " << exists_edge_bitArray << " --updated-> " << exists_edge_bitArray_updated << std::endl;
    
                            switch (exists_edge_bitArray)   // optimisation possible if highest cases are up
                            {
                                case 10:
                                    //three node motif 1 & 11 (0101)   
                                    if(exists_edge_bitArray_updated == 10)
                                        motifStruc.setMotifTypes({1});
                                    else if(exists_edge_bitArray_updated == 58)
                                        motifStruc.setMotifTypes({11});
                                    break;
                                
                                case 5:
                                    //three node motif 3 & 8 (1010)
                                    if(exists_edge_bitArray_updated == 5)
                                        motifStruc.setMotifTypes({3});
                                    else if(exists_edge_bitArray_updated == 53)
                                        motifStruc.setMotifTypes({8});
                                    break;
                                
                                case 6:
                                case 9:
                                    //three node motif 2 & 5 & 7 & 10 (0110, 1001)
                                    if(exists_edge_bitArray_updated == 6 || exists_edge_bitArray_updated == 9)
                                        motifStruc.setMotifTypes({2});
                                    else if(exists_edge_bitArray_updated == 22 || exists_edge_bitArray_updated == 41)
                                        motifStruc.setMotifTypes({5});
                                    else if(exists_edge_bitArray_updated == 38 || exists_edge_bitArray_updated == 25)
                                        motifStruc.setMotifTypes({7});
                                    else if(exists_edge_bitArray_updated == 54 || exists_edge_bitArray_updated == 57)
                                        motifStruc.setMotifTypes({10});
                                    else
                                        std::cout << "error: case 6/9 -> bitArray = " << exists_edge_bitArray_updated << std::endl;
                                    break;
                                
                                case 14:
                                case 11:
                                    //three node motif 4 (0111, 1101)
                                    if(exists_edge_bitArray_updated == 14 || exists_edge_bitArray_updated == 11)
                                        motifStruc.setMotifTypes({4});
                                    break;
                                
                                case 7:
                                case 13:
                                    //three node motif 6 (1110, 1011)
                                    if(exists_edge_bitArray_updated == 7 || exists_edge_bitArray_updated == 13)
                                        motifStruc.setMotifTypes({6});
                                    break;
                                
                                case 15:
                                    //three node motif 9 & 12 & 13 (1111)
                                    if(exists_edge_bitArray_updated == 15)
                                        motifStruc.setMotifTypes({9});
                                    else if(exists_edge_bitArray_updated == 31 || exists_edge_bitArray_updated == 47)
                                        motifStruc.setMotifTypes({12});
                                    else if(exists_edge_bitArray_updated == 63)
                                        motifStruc.setMotifTypes({13});
                                    else
                                        std::cout << "error: case 15 -> bitArray = " << exists_edge_bitArray_updated << std::endl;
                                    break;
                                
                                default:
                                    break;
                            }

                            // Count every motiv
                            //std::cout << "=> motif:  ";
                            for(int motifType = 1; motifType < 14; motifType++) {
                                if(motifStruc.isMotifTypeSet(motifType)) {
                                    //std::cout << motifType;
                                    motifTypeCount[motifType]++;
                                }
                            }
                            //std::cout << "\n" << std::endl;
                        }
                    }
                }
            }
        }
        //Rotational invariant motifs where counted three times each
        if(motifTypeCount[7]%3 != 0) std::cout << "error: motifTypeCount[7]%3 != 0 ==> " << motifTypeCount[7] << std::endl;
        motifTypeCount[7] /= 3;
        if(motifTypeCount[13]%3 != 0) std::cout << "error: motifTypeCount[13]%3 != 0 ==> " << motifTypeCount[13] << std::endl; 
        motifTypeCount[13] /= 3;      
    }
    
    std::vector<long double> motifFraction;
    if(my_rank == result_rank) {
        std::uint64_t total_number_of_motifs = std::accumulate(motifTypeCount.begin(),motifTypeCount.end(), 0);
        motifFraction.resize(motifTypeCount.size());
        motifFraction[0] = total_number_of_motifs;

        for(int motifType = 1; motifType < 14; motifType++) {
            motifFraction[motifType] = motifTypeCount[motifType];
            //motifFraction[motifType] = static_cast<long double>(motifTypeCountTotal[motifType]) / static_cast<long double>(total_number_of_motifs);
        }

        // Print NetworkMotifs:
        for (size_t i = 0; i < motifFraction.size(); i++) {
            std::cout << "motifFraction[" << i << "] = " << motifFraction[i] << std::endl;
        }
    }

    MPIWrapper::barrier();
    return motifFraction;
}

std::uint16_t NetworkMotifs::update_edge_bitArray
(
    const DistributedGraph& graph,
    std::uint16_t exists_edge_bitArray,
    unsigned int node_2_rank, 
    std::uint64_t node_2_local, 
    unsigned int node_3_rank, 
    std::uint64_t node_3_local
)
{
    const std::vector<OutEdge>& oEdges = graph.get_out_edges(node_2_rank, node_2_local);
    const std::vector<InEdge>& iEdges = graph.get_in_edges(node_2_rank, node_2_local);
    
    std::unordered_map<std::pair<std::uint64_t, std::uint64_t>,
                    std::pair<bool, bool>,
                    StdPair_hash> adjacent_nodes;
    
    for(const OutEdge& oEdge : oEdges)
    {
        std::pair<std::uint64_t, std::uint64_t> node_key(oEdge.target_rank, oEdge.target_id);
        std::pair<bool, bool>& value = adjacent_nodes[node_key];
        assert(!value.first);
        value.first = true;
    }
    for(const InEdge& iEdge : iEdges)
    {
        std::pair<std::uint64_t, std::uint64_t> node_key(iEdge.source_rank, iEdge.source_id);
        std::pair<bool, bool>& value = adjacent_nodes[node_key];
        assert(!value.second);
        value.second = true;
    }
    
    std::pair<std::uint64_t, std::uint64_t> node_3_key(node_3_rank, node_3_local);
    std::pair<bool, bool>& value = adjacent_nodes[node_3_key];
    
    bool exists_edge_node2_to_node3 = value.first;
    bool exists_edge_node3_to_node2 = value.second;

    exists_edge_bitArray |= exists_edge_node2_to_node3 ? 16 : 0;
    exists_edge_bitArray |= exists_edge_node3_to_node2 ? 32 : 0;

    // Print decission:
    //std::cout << "edge 2 -> 3: " << exists_edge_node2_to_node3 << "\n" << "edge 3 -> 2: " << exists_edge_node3_to_node2 << std::endl;
    
    return exists_edge_bitArray;
}