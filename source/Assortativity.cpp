#include "Assortativity.h"

std::tuple<double,double,double,double> Assortativity::compute_assortativity_coefficient
(
    const DistributedGraph& graph
)
{
    std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> normalized_node_degree_distribution = compute_normalized_node_degree_distribution(graph);
    
    std::pair<double,double> std_deviation = compute_standard_deviation_of_node_degree_distribution(normalized_node_degree_distribution);
    
    std::unique_ptr<std::tuple<Distribution2D,Distribution2D,Distribution2D,Distribution2D>>joint_edge_degree_distribution=compute_joint_edge_degree_distribution(graph);

    std::tuple<double,double,double,double> result;
    
    //r_In_Out
    Distribution2D& e_in_in = std::get<0>(*joint_edge_degree_distribution);
    Distribution2D& e_in_out = std::get<1>(*joint_edge_degree_distribution);
    Distribution2D& e_out_in = std::get<2>(*joint_edge_degree_distribution);
    Distribution2D& e_out_out = std::get<3>(*joint_edge_degree_distribution);
    std::vector<double>& q_in = normalized_node_degree_distribution->second;
    std::vector<double>& q_out = normalized_node_degree_distribution->first;
    double std_dev_out = std_deviation.first;
    double std_dev_in = std_deviation.second;
    
    assert(q_in.size()==e_in_in.get_first_dimension());
    assert(q_in.size()==e_in_in.get_second_dimension());
    double& r_in_in = std::get<0>(result);
    for(int j=0;j<q_in.size();j++)
    {
        for(int k=0;k<q_in.size();k++)
        {
            r_in_in += j*k*(e_in_in.get_probability(j,k)-q_in[j]*q_in[k]);
        }
    }
    r_in_in/=(std_dev_in*std_dev_in);
    
    assert(q_in.size()==e_in_out.get_first_dimension());
    assert(q_out.size()==e_in_out.get_second_dimension());
    double& r_in_out = std::get<1>(result);
    for(int j=0;j<q_in.size();j++)
    {
        for(int k=0;k<q_out.size();k++)
        {
            r_in_out += j*k*(e_in_out.get_probability(j,k)-q_in[j]*q_out[k]);
        }
    }
    r_in_out/=(std_dev_in*std_dev_out);
    
    assert(q_out.size()==e_out_in.get_first_dimension());
    assert(q_in.size()==e_out_in.get_second_dimension());
    double& r_out_in = std::get<2>(result);
    for(int j=0;j<q_out.size();j++)
    {
        for(int k=0;k<q_in.size();k++)
        {
            r_out_in += j*k*(e_out_in.get_probability(j,k)-q_out[j]*q_in[k]);
        }
    }
    r_out_in/=(std_dev_out*std_dev_in);
    
    assert(q_out.size()==e_out_out.get_first_dimension());
    assert(q_out.size()==e_out_out.get_second_dimension());
    double& r_out_out = std::get<3>(result);
    for(int j=0;j<q_out.size();j++)
    {
        for(int k=0;k<q_out.size();k++)
        {
            r_out_out += j*k*(e_out_out.get_probability(j,k)-q_out[j]*q_out[k]);
        }
    }
    r_out_out/=(std_dev_out*std_dev_out);

    return result;
}

std::pair<int,int> Assortativity::compute_max_nodeDegree_OutIn
(
    const DistributedGraph& graph
)
{
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    const int my_rank = MPIWrapper::get_my_rank();

    int max_out_degree = 0;
    int max_in_degree = 0;
    for(std::uint64_t node_localID=0; node_localID<number_of_ranks; node_localID++)
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
    
    return {global_max_out_degree,global_max_in_degree};
}

std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> Assortativity::compute_node_degree_distribution
(
    const DistributedGraph& graph
)
{
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    const int my_rank = MPIWrapper::get_my_rank();

    std::pair<int,int> max_nodeDegree_InOut = compute_max_nodeDegree_OutIn(graph);
    int global_max_out_degree = max_nodeDegree_InOut.first+1;
    int global_max_in_degree = max_nodeDegree_InOut.second+1;
    
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

std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> Assortativity::compute_normalized_node_degree_distribution
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

std::pair<double,double> Assortativity::compute_standard_deviation_of_node_degree_distribution
(
    const std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>>& node_degree_distribution
)
{
    int node_degree;
    double* result;
    
// Compute expectation
    std::pair<double,double> mean_OutIn = {0,0};    
    auto mean_func=[&](double probability){(*result)+=node_degree*probability; node_degree++;};

    node_degree=0;
    result = &(mean_OutIn.first);
    std::for_each(node_degree_distribution->first.begin(),node_degree_distribution->first.end(),mean_func);    
    node_degree=0;
    result = &(mean_OutIn.second);
    std::for_each(node_degree_distribution->second.begin(),node_degree_distribution->second.end(),mean_func);
    

//Compute variance
    double mean;
    std::pair<double,double> stdDev_OutIn = {0,0};
    auto stdDev_func=[&](double probability){double red=node_degree-mean; (*result)+=red*red*probability; node_degree++;};

    node_degree=0;
    result = &(stdDev_OutIn.first);
    mean = mean_OutIn.first;
    std::for_each(node_degree_distribution->first.begin(),node_degree_distribution->first.end(),stdDev_func);    
    node_degree=0;
    result = &(stdDev_OutIn.second);
    mean = mean_OutIn.second;
    std::for_each(node_degree_distribution->second.begin(),node_degree_distribution->second.end(),stdDev_func);
    
    return stdDev_OutIn;
}

std::unique_ptr<std::tuple<
    Assortativity::Distribution2D,
    Assortativity::Distribution2D,
    Assortativity::Distribution2D,
    Assortativity::Distribution2D>>
Assortativity::compute_joint_edge_degree_distribution
(
    const DistributedGraph& graph
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    
    std::pair<int,int> max_nodeDegree_InOut = compute_max_nodeDegree_OutIn(graph);
    int max_out_degree = max_nodeDegree_InOut.first+1;
    int max_in_degree = max_nodeDegree_InOut.second+1;
    
    std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,EdgeDegrees>>>
        (const DistributedGraph&, std::uint64_t)>
        transfer_edgeDegree_sources = [&](const DistributedGraph& dg,std::uint64_t node_local_ind)
        {
            const std::vector<OutEdge>& oEdges = dg.get_out_edges(my_rank,node_local_ind);
            const std::vector<InEdge>& iEdges = dg.get_in_edges(my_rank,node_local_ind);            
            EdgeDegrees source_edgeDegrees = {oEdges.size(),iEdges.size(),0,0};
            
            auto edgeDegree_sources= std::make_unique<std::vector<std::tuple<std::uint64_t,std::uint64_t,EdgeDegrees>>>(oEdges.size());
            std::transform(oEdges.cbegin(),oEdges.cend(),edgeDegree_sources->begin(),
                           [=](const OutEdge& oEdge)
                                {return std::tie(oEdge.target_rank,oEdge.target_id,source_edgeDegrees);});
            return std::move(edgeDegree_sources);
        };
    std::function<EdgeDegrees(const DistributedGraph&, std::uint64_t, EdgeDegrees)> 
        fill_edgeDegree_target = 
        [&](const DistributedGraph& dg,std::uint64_t node_local_ind,EdgeDegrees source_edgeDegrees)
        {
            const std::vector<OutEdge>& oEdges = dg.get_out_edges(my_rank,node_local_ind);
            const std::vector<InEdge>& iEdges = dg.get_in_edges(my_rank,node_local_ind);
            
            source_edgeDegrees.target_OutDegree = oEdges.size();
            source_edgeDegrees.target_InDegree = iEdges.size();
            return source_edgeDegrees;
        };
    std::unique_ptr<CommunicationPatterns::NodeToNodeQuestionStructure<EdgeDegrees,EdgeDegrees>>
    edge_degrees_OutIn_count=
        CommunicationPatterns::node_to_node_question<EdgeDegrees,EdgeDegrees>
            (graph,MPIWrapper::MPI_EdgeDegrees,transfer_edgeDegree_sources,
             MPIWrapper::MPI_EdgeDegrees,fill_edgeDegree_target);
    
    std::vector<std::uint64_t> inDeg_to_inDeg(max_in_degree*max_in_degree,0);
    std::vector<std::uint64_t> inDeg_to_outDeg(max_in_degree*max_out_degree,0);
    std::vector<std::uint64_t> outDeg_to_inDeg(max_out_degree*max_in_degree,0);
    std::vector<std::uint64_t> outDeg_to_outDeg(max_out_degree*max_out_degree,0);
    
    for(std::uint64_t node_ID = 0;node_ID<number_local_nodes;node_ID++)
    {
        std::unique_ptr<std::vector<EdgeDegrees>> edge_degree_list =
            edge_degrees_OutIn_count->getAnswersOfQuestionerNode(node_ID);
        for(EdgeDegrees& one_edge_degree : *edge_degree_list)
        {
            int inDeg_to_inDeg_index = one_edge_degree.source_InDegree*max_in_degree + one_edge_degree.target_InDegree;
            inDeg_to_inDeg[inDeg_to_inDeg_index]++;
            
            int inDeg_to_outDeg_index = one_edge_degree.source_InDegree*max_in_degree + one_edge_degree.target_OutDegree;
            inDeg_to_outDeg[inDeg_to_outDeg_index]++;
            
            int outDeg_to_inDeg_index = one_edge_degree.source_OutDegree*max_out_degree + one_edge_degree.target_InDegree;
            outDeg_to_inDeg[outDeg_to_inDeg_index]++;
            
            int outDeg_to_outDeg_index = one_edge_degree.source_OutDegree*max_out_degree + one_edge_degree.target_OutDegree;
            outDeg_to_outDeg[outDeg_to_outDeg_index]++;
        }
    }
    
    std::vector<std::uint64_t> global_inDeg_to_inDeg(max_in_degree*max_in_degree,0);
    std::vector<std::uint64_t> global_inDeg_to_outDeg(max_in_degree*max_out_degree,0);
    std::vector<std::uint64_t> global_outDeg_to_inDeg(max_out_degree*max_in_degree,0);
    std::vector<std::uint64_t> global_outDeg_to_outDeg(max_out_degree*max_out_degree,0);
    
    MPIWrapper::all_reduce<std::uint64_t>(inDeg_to_inDeg.data(),global_inDeg_to_inDeg.data(),                                           
                                          inDeg_to_inDeg.size(),MPI_UINT64_T,MPI_SUM);
    MPIWrapper::all_reduce<std::uint64_t>(inDeg_to_outDeg.data(),global_inDeg_to_outDeg.data(),                                           
                                          inDeg_to_outDeg.size(),MPI_UINT64_T,MPI_SUM);
    MPIWrapper::all_reduce<std::uint64_t>(outDeg_to_inDeg.data(),global_outDeg_to_inDeg.data(),                                           
                                          outDeg_to_inDeg.size(),MPI_UINT64_T,MPI_SUM);
    MPIWrapper::all_reduce<std::uint64_t>(outDeg_to_outDeg.data(),global_outDeg_to_outDeg.data(),                                           
                                          outDeg_to_outDeg.size(),MPI_UINT64_T,MPI_SUM);
    
    std::uint64_t sum_global_inDeg_to_inDeg = std::reduce(global_inDeg_to_inDeg.begin(),global_inDeg_to_inDeg.end());
    std::uint64_t sum_global_inDeg_to_outDeg = std::reduce(global_inDeg_to_outDeg.begin(),global_inDeg_to_outDeg.end());
    std::uint64_t sum_global_outDeg_to_inDeg = std::reduce(global_outDeg_to_inDeg.begin(),global_outDeg_to_inDeg.end());
    std::uint64_t sum_global_outDeg_to_outDeg = std::reduce(global_outDeg_to_outDeg.begin(),global_outDeg_to_outDeg.end());

    auto result = std::make_unique<std::tuple<Distribution2D,Distribution2D,Distribution2D,Distribution2D>>();
    
    Distribution2D& inDeg_to_inDeg_distribution = std::get<0>(*result);
    inDeg_to_inDeg_distribution.reset(max_in_degree,max_in_degree);
    
    Distribution2D& inDeg_to_outDeg_distribution = std::get<1>(*result);
    inDeg_to_outDeg_distribution.reset(max_in_degree,max_out_degree);
    
    Distribution2D& outDeg_to_inDeg_distribution = std::get<2>(*result);
    outDeg_to_inDeg_distribution.reset(max_out_degree,max_in_degree);

    Distribution2D& outDeg_to_outDeg_distribution = std::get<3>(*result);
    outDeg_to_outDeg_distribution.reset(max_out_degree,max_out_degree);
    
    for(int i=0;i<global_inDeg_to_inDeg.size();i++)
    {
        int first_index = i%max_in_degree;
        int second_index = i - (i%max_in_degree)*max_in_degree;
        inDeg_to_inDeg_distribution.set_probability(first_index,second_index,static_cast<double>(global_inDeg_to_inDeg[i])/sum_global_inDeg_to_inDeg);
    }
    for(int i=0;i<global_inDeg_to_outDeg.size();i++)
    {
        int first_index = i%max_in_degree;
        int second_index = i - (i%max_in_degree)*max_in_degree;
        inDeg_to_outDeg_distribution.set_probability(first_index,second_index,static_cast<double>(global_inDeg_to_outDeg[i])/sum_global_inDeg_to_outDeg);
    }
    for(int i=0;i<global_outDeg_to_inDeg.size();i++)
    {
        int first_index = i%max_out_degree;
        int second_index = i - (i%max_out_degree)*max_out_degree;
        outDeg_to_inDeg_distribution.set_probability(first_index,second_index,static_cast<double>(global_outDeg_to_inDeg[i])/sum_global_outDeg_to_inDeg);
    }
    for(int i=0;i<global_outDeg_to_outDeg.size();i++)
    {
        int first_index = i%max_out_degree;
        int second_index = i - (i%max_out_degree)*max_out_degree;
        outDeg_to_outDeg_distribution.set_probability(first_index,second_index,static_cast<double>(global_outDeg_to_outDeg[i])/sum_global_outDeg_to_outDeg);
    }
    
    return std::move(result);
}

Assortativity::Distribution2D::Distribution2D()
:
first_dimension(-1),
second_dimension(-1)
{}

Assortativity::Distribution2D::Distribution2D
(
    int first_dimension,
    int second_dimension
)
:
first_dimension(first_dimension),
second_dimension(second_dimension)
{
    probabilities.resize(first_dimension,std::vector<double>(second_dimension,0));
}

void Assortativity::Distribution2D::reset
(
    int first_dimension,
    int second_dimension
)
{
    this->first_dimension = first_dimension;
    this->second_dimension = second_dimension;
    probabilities.resize(first_dimension,std::vector<double>(second_dimension,0));
}


void Assortativity::Distribution2D::set_probability
(
    int first_index,
    int second_index,
    double probability
)
{
    probabilities[first_index][second_index] = probability;
}

double Assortativity::Distribution2D::get_probability
(
    int first_index,
    int second_index
)
{
    return probabilities[first_index][second_index];
}
