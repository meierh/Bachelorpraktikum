#include "Assortativity.h"

std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> compute_node_degree_distribution
(
    const DistributedGraph& graph
)
{
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    const int my_rank = MPIWrapper::get_my_rank();

    int max_out_degree = 0;
    int max_in_degree = 0;
    for(std::uint64_t node_localID=0; node_localID<number_of_ranks;node_localID++)
    {
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_localID);
        const std::vector<InEdge>& iEdges = graph.get_in_edges(my_rank,node_localID);
        
        max_out_degree = std::max<int>(max_out_degree,oEdges.size());
        max_in_degree = std::max<int>(max_in_degree,iEdges.size());
    }
    int global_max_out_degree = 0;
    int global_max_in_degree = 0;
    MPIWrapper::all_reduce<int>(&max_out_degree,&global_max_out_degree,1,MPI_INT,MPI_MAX);
    MPIWrapper::all_reduce<int>(&max_in_degree, &global_max_in_degree, 1,MPI_INT,MPI_MAX);
    
    std::vector<std::uint64_t> out_degree_count(global_max_out_degree);
    std::vector<std::uint64_t> in_degree_count(global_max_in_degree);
    for(std::uint64_t node_localID=0; node_localID<number_of_ranks;node_localID++)
    {
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_localID);
        const std::vector<InEdge>& iEdges = graph.get_in_edges(my_rank,node_localID);
        
        out_degree_count[oEdges.size()]++;
        in_degree_count[iEdges.size()]++;
    }
    std::vector<std::uint64_t> global_out_degree_count(global_max_out_degree);
    std::vector<std::uint64_t> global_in_degree_count(global_max_in_degree);
    
    MPIWrapper::all_reduce<std::uint64_t>(out_degree_count.data(),global_out_degree_count.data(),                                           
                                          global_max_out_degree,MPI_UINT64_T,MPI_SUM);
    MPIWrapper::all_reduce<std::uint64_t>(in_degree_count.data(),global_in_degree_count.data(),                                           
                                          global_max_in_degree,MPI_UINT64_T,MPI_SUM);
    uint64_t sum_global_out_degree_count = std::reduce(global_out_degree_count.cbegin(),global_out_degree_count.cend(),0);
    uint64_t sum_global_in_degree_count = std::reduce(global_in_degree_count.cbegin(),global_in_degree_count.cend(),0);
    
    auto node_degree_distribution = std::make_unique<std::pair<std::vector<double>,std::vector<double>>>();
    node_degree_distribution->first.resize(global_max_out_degree);
    node_degree_distribution->second.resize(global_max_in_degree);
    
    for(int i=0;i<global_max_out_degree;i++)
    {
        node_degree_distribution->first[i] = static_cast<double>(global_out_degree_count[i])/sum_global_out_degree_count;
    }
    for(int i=0;i<global_max_in_degree;i++)
    {
        node_degree_distribution->second[i] = static_cast<double>(global_in_degree_count[i])/sum_global_in_degree_count;
    }
    return node_degree_distribution;
}

std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> compute_normalized_node_degree_distribution
(
    const DistributedGraph& graph
)
{
    std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> node_degree_distribution;
    node_degree_distribution = compute_node_degree_distribution(graph);
    
    double expectation_out_degree_distribution=0;
    int j=0;
    std::for_each(node_degree_distribution->first.cbegin(),node_degree_distribution->first.cend(),
                  [&](double p){expectation_out_degree_distribution+=j*p; j++;});
    
    double expectation_in_degree_distribution=0;
    j=0;
    std::for_each(node_degree_distribution->second.cbegin(),node_degree_distribution->second.cend(),
                  [&](double p){expectation_in_degree_distribution+=j*p; j++;});

    
    auto normalized_node_degree_distribution = std::make_unique<std::pair<std::vector<double>,std::vector<double>>>();
    node_degree_distribution->first.resize(node_degree_distribution->first.size()-1);
    node_degree_distribution->second.resize(node_degree_distribution->second.size()-1);
    
    int k=0;
    std::transform(node_degree_distribution->first.cbegin()+1,node_degree_distribution->first.cend(),
                   normalized_node_degree_distribution->first.begin(),
                   [&](double p){return (k+1)*p;});
    
    k=0;
    std::transform(node_degree_distribution->second.cbegin()+1,node_degree_distribution->second.cend(),
                   normalized_node_degree_distribution->second.begin(),
                   [&](double p){return (k+1)*p;});
    
    return normalized_node_degree_distribution;
}
