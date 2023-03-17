#include "Assortativity.h"

std::tuple<double,double,double,double> Assortativity::compute_assortativity_coefficient
(
    const DistributedGraph& graph
)
{
    const int my_rank = MPIWrapper::get_my_rank();

    std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> normalized_node_degree_distribution = compute_normalized_node_degree_distribution(graph);
    
    std::pair<double,double> std_deviation = compute_standard_deviation_of_node_degree_distribution(normalized_node_degree_distribution);
    
    std::unique_ptr<std::tuple<Distribution2D<double>,Distribution2D<double>,Distribution2D<double>,Distribution2D<double>>>joint_edge_degree_distribution=compute_joint_edge_degree_distribution(graph);

    std::tuple<double,double,double,double> result;
    
    //r_In_Out
    Distribution2D<double>& e_in_in = std::get<0>(*joint_edge_degree_distribution);
    Distribution2D<double>& e_in_out = std::get<1>(*joint_edge_degree_distribution);
    Distribution2D<double>& e_out_in = std::get<2>(*joint_edge_degree_distribution);
    Distribution2D<double>& e_out_out = std::get<3>(*joint_edge_degree_distribution);
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
    if(std_dev_in!=0)
    {
        r_in_in/=(std_dev_in*std_dev_in);
    }
    
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
    if(std_dev_in!=0 && std_dev_out!=0)
    {
        r_in_out/=(std_dev_in*std_dev_out);
    }
    
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
    if(std_dev_in!=0 && std_dev_out!=0)
    {
        r_out_in/=(std_dev_out*std_dev_in);
    }
    
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
    if(std_dev_out!=0)
    {
        r_out_out/=(std_dev_out*std_dev_out);
    }
    
    
    /*if(my_rank==0)
        std::cout << "par results:\nr_in_in = " << r_in_in << "\nr_in_out = " << r_in_out 
                << "\nr_out_in = " << r_out_in << "\nr_out_out = " << r_out_out << std::endl;*/
                

    return result;
}

std::pair<std::uint64_t,std::uint64_t> Assortativity::compute_max_nodeDegree_OutIn
(
    const DistributedGraph& graph
)
{
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    const int my_rank = MPIWrapper::get_my_rank();
	const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    
    std::uint64_t max_out_degree = 0;
    std::uint64_t max_in_degree = 0;
    for(std::uint64_t node_localID=0; node_localID<number_local_nodes; node_localID++)
    {
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_localID);
        const std::vector<InEdge>& iEdges = graph.get_in_edges(my_rank,node_localID);
        
        max_out_degree = std::max<std::uint64_t>(max_out_degree,static_cast<std::uint64_t>(oEdges.size()));
        max_in_degree = std::max<std::uint64_t>(max_in_degree,static_cast<std::uint64_t>(iEdges.size()));
    }
        
    std::uint64_t global_max_out_degree = 0;
    std::uint64_t global_max_in_degree = 0;
    MPIWrapper::all_reduce<std::uint64_t>(&max_out_degree,&global_max_out_degree,1,MPI_UINT64_T,MPI_MAX);
    MPIWrapper::all_reduce<std::uint64_t>(&max_in_degree, &global_max_in_degree, 1,MPI_UINT64_T,MPI_MAX);
    
    return {global_max_out_degree,global_max_in_degree};
}

std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> Assortativity::compute_node_degree_distribution
(
    const DistributedGraph& graph
)
{    
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    const int my_rank = MPIWrapper::get_my_rank();
	const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    
    std::pair<int,int> max_nodeDegree_OutIn = compute_max_nodeDegree_OutIn(graph);
    
    int global_max_out_degree = max_nodeDegree_OutIn.first+1;
    int global_max_in_degree = max_nodeDegree_OutIn.second+1;
    
    std::vector<std::uint64_t> out_degree_count(global_max_out_degree);
    std::vector<std::uint64_t> in_degree_count(global_max_in_degree);
    for(std::uint64_t node_localID=0; node_localID<number_local_nodes;node_localID++)
    {
        const std::vector<OutEdge>& oEdges = graph.get_out_edges(my_rank,node_localID);
        const std::vector<InEdge>& iEdges = graph.get_in_edges(my_rank,node_localID);
        
        assert(oEdges.size()<out_degree_count.size());            assert(iEdges.size()<in_degree_count.size());
        
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
    
    return std::move(node_degree_distribution);
}

std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> Assortativity::compute_normalized_node_degree_distribution
(
    const DistributedGraph& graph
)
{
    const int number_of_ranks = MPIWrapper::get_number_ranks();
    const int my_rank = MPIWrapper::get_my_rank();
    
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
    normalized_node_degree_distribution->first.resize(node_degree_distribution->first.size()-1);
    normalized_node_degree_distribution->second.resize(node_degree_distribution->second.size()-1);
       
    double* expectation; 
    int k;
    auto normalization = [&](double p){double q_k=(k+1)*p/(*expectation);k++;return q_k;};    
    
    k=0;
    expectation = &expectation_out_degree_distribution;
    std::transform(node_degree_distribution->first.cbegin()+1,node_degree_distribution->first.cend(),
                   normalized_node_degree_distribution->first.begin(),normalization);
    
    k=0;
    expectation = &expectation_in_degree_distribution;
    std::transform(node_degree_distribution->second.cbegin()+1,node_degree_distribution->second.cend(),
                   normalized_node_degree_distribution->second.begin(),normalization);
    
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
    *result = std::sqrt(*result);
    
    node_degree=0;
    result = &(stdDev_OutIn.second);
    mean = mean_OutIn.second;
    std::for_each(node_degree_distribution->second.begin(),node_degree_distribution->second.end(),stdDev_func);
    *result = std::sqrt(*result);
    
    return stdDev_OutIn;
}

std::unique_ptr<std::tuple<
    Assortativity::Distribution2D<double>,
    Assortativity::Distribution2D<double>,
    Assortativity::Distribution2D<double>,
    Assortativity::Distribution2D<double>>>
Assortativity::compute_joint_edge_degree_distribution
(
    const DistributedGraph& graph
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    
    std::pair<int,int> max_nodeDegree_OutIn = compute_max_nodeDegree_OutIn(graph);
    int max_out_degree = max_nodeDegree_OutIn.first;
    int max_in_degree = max_nodeDegree_OutIn.second; 
    
    std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,EdgeDegrees>>>
        (const DistributedGraph&, std::uint64_t)>
        transfer_edgeDegree_sources = [&](const DistributedGraph& dg,std::uint64_t node_local_ind)
        {
		    const std::vector<OutEdge>& oEdges = dg.get_out_edges(my_rank, node_local_ind);
            std::uint64_t numOut = dg.get_number_out_edges(my_rank,node_local_ind);
            std::uint64_t numIn = dg.get_number_in_edges(my_rank,node_local_ind);
                        
            EdgeDegrees source_edgeDegrees;
            source_edgeDegrees.source_OutDegree = static_cast<std::uint64_t>(numOut);
            source_edgeDegrees.source_InDegree = static_cast<std::uint64_t>(numIn);
                        
            auto edgeDegree_sources= std::make_unique<std::vector<std::tuple<std::uint64_t,std::uint64_t,EdgeDegrees>>>(oEdges.size());
            std::transform(oEdges.cbegin(),oEdges.cend(),edgeDegree_sources->begin(),
                           [=](const OutEdge& oEdge)
                           {
                               return std::tuple<std::uint64_t,std::uint64_t,EdgeDegrees>(oEdge.target_rank,oEdge.target_id,source_edgeDegrees);
                           });
            return std::move(edgeDegree_sources);
        };
    std::function<EdgeDegrees(const DistributedGraph&, std::uint64_t, EdgeDegrees)> 
        fill_edgeDegree_target = 
        [&](const DistributedGraph& dg,std::uint64_t node_local_ind,EdgeDegrees source_edgeDegrees)
        {
            std::uint64_t numOut = dg.get_number_out_edges(my_rank,node_local_ind);
            std::uint64_t numIn = dg.get_number_in_edges(my_rank,node_local_ind);
            
            source_edgeDegrees.target_OutDegree = static_cast<std::uint64_t>(numOut);
            source_edgeDegrees.target_InDegree = static_cast<std::uint64_t>(numIn);
            
            return source_edgeDegrees;
        };
    std::unique_ptr<CommunicationPatterns::NodeToNodeQuestionStructure<EdgeDegrees,EdgeDegrees>>
    edge_degrees_OutIn_count=
        CommunicationPatterns::node_to_node_question<EdgeDegrees,EdgeDegrees>
            (graph,MPIWrapper::MPI_EdgeDegrees,transfer_edgeDegree_sources,
             MPIWrapper::MPI_EdgeDegrees,fill_edgeDegree_target);  
    
    Distribution2D<std::uint64_t> inDeg_to_inDeg_count(max_in_degree+1,max_in_degree+1);
    Distribution2D<std::uint64_t> inDeg_to_outDeg_count(max_in_degree+1,max_out_degree+1);
    Distribution2D<std::uint64_t> outDeg_to_inDeg_count(max_out_degree+1,max_in_degree+1);
    Distribution2D<std::uint64_t> outDeg_to_outDeg_count(max_out_degree+1,max_out_degree+1);
    
    for(std::uint64_t node_ID = 0;node_ID<number_local_nodes;node_ID++)
    {
        std::unique_ptr<std::vector<EdgeDegrees>> edge_degree_list =
            edge_degrees_OutIn_count->getAnswersOfQuestionerNode(node_ID);
        for(EdgeDegrees& one_edge_degree : *edge_degree_list)
        {
            auto increment_by_one = [](std::uint64_t value){return value+1;};
            
            inDeg_to_inDeg_count.operate_on_index(one_edge_degree.source_InDegree,
                                                one_edge_degree.target_InDegree,increment_by_one);
            
            inDeg_to_outDeg_count.operate_on_index(one_edge_degree.source_InDegree,
                                                one_edge_degree.target_OutDegree,increment_by_one);
            
            outDeg_to_inDeg_count.operate_on_index(one_edge_degree.source_OutDegree,
                                                one_edge_degree.target_InDegree,increment_by_one);
            
            outDeg_to_outDeg_count.operate_on_index(one_edge_degree.source_OutDegree,
                                                one_edge_degree.target_OutDegree,increment_by_one);
        }
    }
    
    Distribution2D<std::uint64_t> glob_inDeg_to_inDeg_count(max_in_degree+1,max_in_degree+1);
    Distribution2D<std::uint64_t> glob_inDeg_to_outDeg_count(max_in_degree+1,max_out_degree+1);
    Distribution2D<std::uint64_t> glob_outDeg_to_inDeg_count(max_out_degree+1,max_in_degree+1);
    Distribution2D<std::uint64_t> glob_outDeg_to_outDeg_count(max_out_degree+1,max_out_degree+1);
    
    MPIWrapper::all_reduce<std::uint64_t>(inDeg_to_inDeg_count.data_ptr(),
                                          glob_inDeg_to_inDeg_count.data_ptr(),                                           
                                          inDeg_to_inDeg_count.data().size(),
                                          MPI_UINT64_T,MPI_SUM);
    MPIWrapper::all_reduce<std::uint64_t>(inDeg_to_outDeg_count.data_ptr(),
                                          glob_inDeg_to_outDeg_count.data_ptr(),                                           
                                          inDeg_to_outDeg_count.data().size(),
                                          MPI_UINT64_T,MPI_SUM);
    MPIWrapper::all_reduce<std::uint64_t>(outDeg_to_inDeg_count.data_ptr(),
                                          glob_outDeg_to_inDeg_count.data_ptr(),                                           
                                          outDeg_to_inDeg_count.data().size(),
                                          MPI_UINT64_T,MPI_SUM);
    MPIWrapper::all_reduce<std::uint64_t>(outDeg_to_outDeg_count.data_ptr(),
                                          glob_outDeg_to_outDeg_count.data_ptr(),                                           
                                          outDeg_to_outDeg_count.data().size(),
                                          MPI_UINT64_T,MPI_SUM);
    
    Distribution2D<double> inDeg_to_inDeg(max_in_degree+1,max_in_degree+1);
    Distribution2D<double> inDeg_to_outDeg(max_in_degree+1,max_out_degree+1);
    Distribution2D<double> outDeg_to_inDeg(max_out_degree+1,max_in_degree+1);
    Distribution2D<double> outDeg_to_outDeg(max_out_degree+1,max_out_degree+1);
    
    std::uint64_t total;
    auto compute_probability = [&](std::uint64_t count){return static_cast<double>(count)/total;};
    
    total = std::reduce(glob_inDeg_to_inDeg_count.data().cbegin(),glob_inDeg_to_inDeg_count.data().cend());
    std::transform(glob_inDeg_to_inDeg_count.data().cbegin(),glob_inDeg_to_inDeg_count.data().cend(),inDeg_to_inDeg.data().begin(),compute_probability);

    total = std::reduce(glob_inDeg_to_outDeg_count.data().cbegin(),glob_inDeg_to_outDeg_count.data().cend());
    std::transform(glob_inDeg_to_outDeg_count.data().cbegin(),glob_inDeg_to_outDeg_count.data().cend(),inDeg_to_outDeg.data().begin(),compute_probability);
    
    total = std::reduce(glob_outDeg_to_inDeg_count.data().cbegin(),glob_outDeg_to_inDeg_count.data().cend());
    std::transform(glob_outDeg_to_inDeg_count.data().cbegin(),glob_outDeg_to_inDeg_count.data().cend(),outDeg_to_inDeg.data().begin(),compute_probability);

    total = std::reduce(glob_outDeg_to_outDeg_count.data().cbegin(),glob_outDeg_to_outDeg_count.data().cend());
    std::transform(glob_outDeg_to_outDeg_count.data().cbegin(),glob_outDeg_to_outDeg_count.data().cend(),outDeg_to_outDeg.data().begin(),compute_probability);

    auto result = std::make_unique<std::tuple<Distribution2D<double>,Distribution2D<double>,Distribution2D<double>,Distribution2D<double>>>();
    
    Distribution2D<double>& normalized_inDeg_to_inDeg = std::get<0>(*result);
    normalized_inDeg_to_inDeg.reset(max_in_degree,max_in_degree);
    
    Distribution2D<double>& normalized_inDeg_to_outDeg = std::get<1>(*result);
    normalized_inDeg_to_outDeg.reset(max_in_degree,max_out_degree);
    
    Distribution2D<double>& normalized_outDeg_to_inDeg = std::get<2>(*result);
    normalized_outDeg_to_inDeg.reset(max_out_degree,max_in_degree);

    Distribution2D<double>& normalized_outDeg_to_outDeg = std::get<3>(*result);
    normalized_outDeg_to_outDeg.reset(max_out_degree,max_out_degree);
    
    double expectation;
    Distribution2D<double>* source_distribution;
    auto compute_normalized_edge_degrees_distribution = [&](int i,int j,double x)
    {
        if(expectation!=0)
        {
            return (i+1)*(j+1)*source_distribution->get_probability(i+1,j+1) / expectation;
        }
        else
        {
            return (i+1)*(j+1)*source_distribution->get_probability(i+1,j+1);
        }
    };
    
    expectation = inDeg_to_inDeg.compute_Expectation();
    source_distribution = &inDeg_to_inDeg;
    normalized_inDeg_to_inDeg.operate_on_all(compute_normalized_edge_degrees_distribution);

    expectation = inDeg_to_outDeg.compute_Expectation();
    source_distribution = &inDeg_to_outDeg;
    normalized_inDeg_to_outDeg.operate_on_all(compute_normalized_edge_degrees_distribution);

    expectation = outDeg_to_inDeg.compute_Expectation();
    source_distribution = &outDeg_to_inDeg;
    normalized_outDeg_to_inDeg.operate_on_all(compute_normalized_edge_degrees_distribution);
    
    expectation = outDeg_to_outDeg.compute_Expectation();
    source_distribution = &outDeg_to_outDeg;
    normalized_outDeg_to_outDeg.operate_on_all(compute_normalized_edge_degrees_distribution);
    
    return std::move(result);
}

template<typename T>
Assortativity::Distribution2D<T>::Distribution2D()
:
first_dimension(-1),
second_dimension(-1)
{}

template<typename T>
Assortativity::Distribution2D<T>::Distribution2D
(
    int first_dimension,
    int second_dimension
)
:
first_dimension(first_dimension),
second_dimension(second_dimension)
{
    probabilities.resize(first_dimension*second_dimension,0);
}

template<typename T>
void Assortativity::Distribution2D<T>::reset
(
    int first_dimension,
    int second_dimension
)
{
    this->first_dimension = first_dimension;
    this->second_dimension = second_dimension;
    probabilities.resize(first_dimension*second_dimension,0);
}

template<typename T>
void Assortativity::Distribution2D<T>::set_probability
(
    int first_index,
    int second_index,
    T probability
)
{
    int index = first_index*second_dimension+second_index;
    if(!(index<probabilities.size()))
        std::cout<<"["<<index<<"|"<<probabilities.size()<<"]"<<std::endl;
    assert(index<probabilities.size());    
    probabilities[index] = probability;
}

template<typename T>
T Assortativity::Distribution2D<T>::get_probability
(
    int first_index,
    int second_index
)
{
    int index = first_index*second_dimension+second_index;
    if(!(index<probabilities.size()))
        std::cout<<"first:"<<first_index<<" second:"<<second_index<<" ["<<index<<"|"<<probabilities.size()<<"]"<<std::endl;
    assert(index<probabilities.size());
    return probabilities[index];
}

template<typename T>
void Assortativity::Distribution2D<T>::operate_on_index
(
    int first_index,
    int second_index,
    std::function<T(T)> operation
)
{
    T new_value = operation(get_probability(first_index,second_index));
    set_probability(first_index,second_index,new_value);
};

template<typename T>
T Assortativity::Distribution2D<T>::compute_Expectation()
{
    T expectation = 0;
    for(int i=0;i<first_dimension;i++)
    {
        for(int j=0;j<second_dimension;j++)
        {
            expectation += i*j*get_probability(i,j);
        }
    }
    return expectation;
}

template<typename T>
void Assortativity::Distribution2D<T>::operate_on_all
(
    std::function<T(int,int,T)> operation
)
{
    for(int i=0;i<first_dimension;i++)
    {
        for(int j=0;j<second_dimension;j++)
        {
            set_probability(i,j,operation(i,j,get_probability(i,j)));
        }
    }
}

std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> Assortativity::compute_node_degree_distribution_sequential
(    
    const DistributedGraph& graph
)
{
    unsigned int result_rank = 0;
    const int number_ranks = MPIWrapper::get_number_ranks();
    const int my_rank = MPIWrapper::get_my_rank();

    std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    
    std::vector<std::uint64_t> number_nodes_of_ranks(number_ranks);
    MPIWrapper::gather<std::uint64_t>(&number_local_nodes, number_nodes_of_ranks.data(), 1, MPI_UINT64_T, result_rank);
    
    auto node_degree_distribution = std::make_unique<std::pair<std::vector<double>,std::vector<double>>>();

    if(my_rank == result_rank) 
    {
        // Determine node degree distributions (p_out, p_in):
        int max_out_degree = 0;
        int max_in_degree = 0;

        for(int current_rank = 0; current_rank < number_ranks; current_rank++) {
            for(std::uint64_t current_node = 0; current_node < number_nodes_of_ranks[current_rank]; current_node++) {
                
                const std::uint64_t number_out_edges = graph.get_number_out_edges(current_rank, current_node);
                const std::uint64_t number_in_edges = graph.get_number_in_edges(current_rank, current_node);

                max_out_degree = (max_out_degree < number_out_edges) ? number_out_edges : max_out_degree;
                max_in_degree = (max_in_degree < number_in_edges) ? number_in_edges : max_in_degree;
            }
        }
        
        std::vector<std::uint64_t> out_degree_count(max_out_degree + 1);
        std::vector<std::uint64_t> in_degree_count(max_in_degree + 1);

        for(int current_rank = 0; current_rank < number_ranks; current_rank++) {
            for(std::uint64_t current_node = 0; current_node < number_nodes_of_ranks[current_rank]; current_node++) {
                const std::vector<OutEdge>& out_edges = graph.get_out_edges(current_rank, current_node);
                const std::vector<InEdge>& in_edges = graph.get_in_edges(current_rank, current_node);
                out_degree_count[out_edges.size()]++;
                in_degree_count[in_edges.size()]++;
            }
        }

        std::uint64_t sum_out_degree_count = std::reduce(out_degree_count.cbegin(), out_degree_count.cend(), 0);
        std::uint64_t sum_in_degree_count = std::reduce(in_degree_count.cbegin(), in_degree_count.cend(), 0);

        std::vector<double> node_degree_distributions_out(out_degree_count.size());
        std::vector<double> node_degree_distributions_in(in_degree_count.size());
        
        for(int i = 0; i < node_degree_distributions_out.size(); i++) {
            node_degree_distributions_out[i] = static_cast<double>(out_degree_count[i]) / sum_out_degree_count;
        }
        
        for(int i = 0; i < node_degree_distributions_in.size(); i++) {
            node_degree_distributions_in[i] = static_cast<double>(in_degree_count[i]) / sum_in_degree_count;
        }
    
        node_degree_distribution->first = std::move(node_degree_distributions_out);
        node_degree_distribution->second = std::move(node_degree_distributions_in);
    }
    
    return std::move(node_degree_distribution);
}

std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> Assortativity::compute_normalized_node_degree_distribution_sequential
(    
    const DistributedGraph& graph
)
{
    unsigned int result_rank = 0;
    const int my_rank = MPIWrapper::get_my_rank();
    
    auto node_degree_distribution = compute_node_degree_distribution_sequential(graph);
    std::vector<double>& node_degree_distributions_out = node_degree_distribution->first;
    std::vector<double>& node_degree_distributions_in = node_degree_distribution->second;
    
    auto normalized_node_degree_distribution = std::make_unique<std::pair<std::vector<double>,std::vector<double>>>();

    if(my_rank == result_rank) 
    {
        // Determine normalized node degree distributions (q_out, q_in):
        double expectation_out_degree_distribution = 0;
        for(int j = 0; j < node_degree_distributions_out.size(); j++){
            expectation_out_degree_distribution += j * node_degree_distributions_out[j];
        }
        //std::cout<<"Seq expectation_out_degree_distribution:"<<expectation_out_degree_distribution<<std::endl;
        
        double expectation_in_degree_distribution = 0;
        for(int j = 0; j < node_degree_distributions_in.size(); j++){
            expectation_in_degree_distribution += j * node_degree_distributions_in[j];
        }
        //std::cout<<"Seq expectation_in_degree_distribution:"<<expectation_in_degree_distribution<<std::endl;

        std::vector<double> q_out(node_degree_distributions_out.size()-1);
        std::vector<double> q_in(node_degree_distributions_in.size()-1);
        for(int k = 0; k < q_out.size(); k++){
            q_out[k] = static_cast<double>((k+1) * node_degree_distributions_out[k+1]) / expectation_out_degree_distribution;
        }
        for(int k = 0; k < q_in.size(); k++){
            q_in[k] = static_cast<double>((k+1) * node_degree_distributions_in[k+1]) / expectation_in_degree_distribution;
        }
        
        normalized_node_degree_distribution->first = std::move(q_out);
        normalized_node_degree_distribution->second = std::move(q_in);
    }
    
    return std::move(normalized_node_degree_distribution);
}

std::pair<double,double> Assortativity::compute_standard_deviation_of_node_degree_distribution_sequential
(
    const std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>>& node_degree_distribution
)
{
    unsigned int result_rank = 0;
    const int my_rank = MPIWrapper::get_my_rank();
    std::pair<double,double> result(0,0);
    
    if(my_rank == result_rank) 
    {
        std::vector<double>& q_out = node_degree_distribution->first;
        std::vector<double>& q_in = node_degree_distribution->second;

        // Determine standard deviation of node degree distribution (std_dev_out, std_dev_in):
        double right_out = 0.0;
        double right_in = 0.0;        
        for(int out_degree = 0; out_degree < q_out.size(); out_degree++){
            right_out += (out_degree * q_out[out_degree]);
        }
        right_out = right_out*right_out;
        for(int in_degree = 0; in_degree < q_in.size(); in_degree++){
            right_in += (in_degree * q_in[in_degree]);
        }
        right_in = right_in*right_in;

        double left_out = 0.0;
        double left_in = 0.0;
        for(int out_degree = 0; out_degree < q_out.size(); out_degree++){
            left_out += (out_degree * out_degree) * q_out[out_degree];
        }
        for(int in_degree = 0; in_degree < q_in.size(); in_degree++){
            left_in += (in_degree * in_degree) * q_in[in_degree];
        }
        
        double std_dev_out = std::sqrt(left_out - right_out);
        double std_dev_in = std::sqrt(left_in - right_in);
        result.first = std_dev_out;
        result.second = std_dev_in;
    }
    return result;
}

std::unique_ptr<std::tuple<
    Assortativity::Distribution2D<double>,
    Assortativity::Distribution2D<double>,
    Assortativity::Distribution2D<double>,
    Assortativity::Distribution2D<double>>>
Assortativity::compute_joint_edge_degree_distribution_sequential
(
    const DistributedGraph& graph
)
{
    unsigned int result_rank = 0;
    const int number_ranks = MPIWrapper::get_number_ranks();
    const int my_rank = MPIWrapper::get_my_rank();

    std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    const auto number_total_nodes = MPIWrapper::all_reduce_sum(number_local_nodes);
    
    std::vector<std::uint64_t> number_nodes_of_ranks(number_ranks);
    MPIWrapper::gather<std::uint64_t>(&number_local_nodes, number_nodes_of_ranks.data(), 1, MPI_UINT64_T, result_rank);
    
    auto result = std::make_unique<std::tuple<Distribution2D<double>,Distribution2D<double>,Distribution2D<double>,Distribution2D<double>>>();
    
    Distribution2D<double>& e_in_in = std::get<0>(*result);
    Distribution2D<double>& e_in_out = std::get<1>(*result);
    Distribution2D<double>& e_out_in = std::get<2>(*result);
    Distribution2D<double>& e_out_out = std::get<3>(*result);

    if(my_rank == result_rank) {

        struct StdPair_hash {
            std::size_t operator () (const std::pair<std::uint64_t, std::uint64_t> &p) const {
                return std::hash<std::uint64_t>{}(p.first) ^ std::hash<std::uint64_t>{}(p.second);
            }
        };

		std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, 
                           std::pair<std::uint64_t, std::uint64_t>, //(number_out_edges, number_in_edges)
                           StdPair_hash>
		    nodeID_to_degrees;

        // Determine node degree distributions (p_out, p_in):
        int max_out_degree = 0;
        int max_in_degree = 0;

        for(int current_rank = 0; current_rank < number_ranks; current_rank++)
        {
            for(std::uint64_t current_node=0;current_node<number_nodes_of_ranks[current_rank];current_node++)
            {
                const std::uint64_t number_out_edges = graph.get_number_out_edges(current_rank, current_node);
                const std::uint64_t number_in_edges = graph.get_number_in_edges(current_rank, current_node);

                std::pair<std::uint64_t, std::uint64_t> key(current_rank, current_node);
                std::pair<std::uint64_t, std::uint64_t>& value = nodeID_to_degrees[key];
                assert(!value.first && !value.second);
                value = {number_out_edges, number_in_edges};

                max_out_degree = (max_out_degree < number_out_edges) ? number_out_edges : max_out_degree;
                max_in_degree = (max_in_degree < number_in_edges) ? number_in_edges : max_in_degree;
            }
        }

        e_in_in.reset(max_in_degree, max_in_degree);
        e_in_out.reset(max_in_degree, max_out_degree);
        e_out_in.reset(max_out_degree, max_in_degree);
        e_out_out.reset(max_out_degree, max_out_degree);
        
        std::uint64_t sum_inDeg_to_inDeg;
        std::uint64_t sum_inDeg_to_outDeg;
        std::uint64_t sum_outDeg_to_inDeg;
        std::uint64_t sum_outDeg_to_outDeg;

        for(int src_r = 0; src_r < number_ranks; src_r++)
        {
            for(std::uint64_t src_n = 0; src_n < number_nodes_of_ranks[src_r]; src_n++)
            {
                std::pair<std::uint64_t, std::uint64_t> src_key(src_r, src_n);
                std::pair<std::uint64_t, std::uint64_t> src_value = nodeID_to_degrees[src_key];
                std::uint64_t src_out_degree = src_value.first-1;
                std::uint64_t src_in_degree = src_value.second-1;

                const std::vector<OutEdge>& out_edges = graph.get_out_edges(src_r, src_n);
                for(const OutEdge& out_edge : out_edges) {

                    std::pair<std::uint64_t, std::uint64_t> dest_key(out_edge.target_rank, out_edge.target_id);
                    std::pair<std::uint64_t, std::uint64_t> dest_value = nodeID_to_degrees[dest_key];
                    
                    std::uint64_t dest_out_degree = dest_value.first-1;
                    std::uint64_t dest_in_degree = dest_value.second-1;
                    
                    e_in_in.set_probability(src_in_degree, dest_in_degree, e_in_in.get_probability(src_in_degree, dest_in_degree)+1);
                    sum_inDeg_to_inDeg++;

                    e_in_out.set_probability(src_in_degree, dest_out_degree, e_in_out.get_probability(src_in_degree, dest_out_degree)+1);
                    sum_inDeg_to_outDeg++;

                    e_out_in.set_probability(src_out_degree, dest_in_degree, e_out_in.get_probability(src_out_degree, dest_in_degree)+1);
                    sum_outDeg_to_inDeg++;

                    e_out_out.set_probability(src_out_degree, dest_out_degree, e_out_out.get_probability(src_out_degree, dest_out_degree)+1);
                    sum_outDeg_to_outDeg++;
                }
            }
        }

        for(int src = 0; src < max_in_degree; src++){
            for(int tar = 0; tar < max_in_degree; tar++){
                e_in_in.set_probability(src, tar, static_cast<double>(e_in_in.get_probability(src, tar))/sum_inDeg_to_inDeg);
            }
        }

        for(int src = 0; src < max_in_degree; src++){
            for(int tar = 0; tar < max_out_degree; tar++){
                e_in_out.set_probability(src, tar, static_cast<double>(e_in_out.get_probability(src, tar))/sum_inDeg_to_outDeg);
            }
        }

        for(int src = 0; src < max_out_degree; src++){
            for(int tar = 0; tar < max_in_degree; tar++){
                e_out_in.set_probability(src, tar, static_cast<double>(e_out_in.get_probability(src, tar))/sum_outDeg_to_inDeg);
            }
        }

        for(int src = 0; src < max_out_degree; src++){
            for(int tar = 0; tar < max_out_degree; tar++){
                e_out_out.set_probability(src, tar, static_cast<double>(e_out_out.get_probability(src, tar))/sum_outDeg_to_outDeg);
            }
        }
    }
    
    return std::move(result); 
}

std::tuple<double,double,double,double> Assortativity::compute_assortativity_coefficient_sequential
(
    const DistributedGraph& graph
)
{
    unsigned int result_rank = 0;
    const int number_ranks = MPIWrapper::get_number_ranks();
    const int my_rank = MPIWrapper::get_my_rank();

    std::uint64_t number_local_nodes = graph.get_number_local_nodes();
    const auto number_total_nodes = MPIWrapper::all_reduce_sum(number_local_nodes);
    
    std::vector<std::uint64_t> number_nodes_of_ranks(number_ranks);
    MPIWrapper::gather<std::uint64_t>(&number_local_nodes, number_nodes_of_ranks.data(), 1, MPI_UINT64_T, result_rank);

    std::tuple<double, double, double, double> result;

    if(my_rank == result_rank) 
    {
        auto normalized_distribution = compute_normalized_node_degree_distribution_sequential(graph);

        std::pair<double,double>  std_dev = compute_standard_deviation_of_node_degree_distribution_sequential(normalized_distribution);
        double std_dev_out = std_dev.first;
        double std_dev_in = std_dev.second;

        std::vector<double> q_out;
        std::vector<double> q_in;
        q_out = std::move(normalized_distribution->first);
        q_in = std::move(normalized_distribution->second);


        std::unique_ptr<std::tuple<Distribution2D<double>,Distribution2D<double>,Distribution2D<double>,Distribution2D<double>>>joint_edge_degree_distribution=compute_joint_edge_degree_distribution_sequential(graph);

        std::tuple<double,double,double,double> result;
    
        //r_In_Out
        Distribution2D<double>& e_in_in = std::get<0>(*joint_edge_degree_distribution);
        Distribution2D<double>& e_in_out = std::get<1>(*joint_edge_degree_distribution);
        Distribution2D<double>& e_out_in = std::get<2>(*joint_edge_degree_distribution);
        Distribution2D<double>& e_out_out = std::get<3>(*joint_edge_degree_distribution);
        
        // Determine Assortativities (r_in_in, r_in_out, r_out_in, r_out_out):
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

        // Print results:
        
        std::cout << "seq results:\nr_in_in = " << r_in_in << "\nr_in_out = " << r_in_out 
                  << "\nr_out_in = " << r_out_in << "\nr_out_out = " << r_out_out << std::endl;
                  
        
        // Maybe add broadcast in MPIWrapper and use it here for valid result in all ranks
    }
    
    return result;
}

void compareDistributions
(
    const std::vector<double>& distrA,
    const std::vector<double>& distrB,
    std::string text
)
{
    if(distrA.size()!=distrB.size())
    {
        std::cout<<text<<"  size: A "<<distrA.size()<<" B "<<distrB.size()<<std::endl;
        assert(false);
    }
    double rel_error_sum = 0;
    for(int i=0;i<distrB.size();i++)
    {
        double abs_base = 0.5*(distrB[i]+distrA[i]);
        double abs_error = std::abs(distrB[i]-distrA[i]);
        double rel_error;
        if(abs_error==0)
            rel_error = 0;
        else if(abs_base==0)
            rel_error = abs_error;
        else
            rel_error = abs_error/abs_base;
        
        if(rel_error>1e-10)
        {
            std::cout<<text<<"  bin:"<<i<<" has rel_error:"<<rel_error<<" A[]:"<<distrA[i]<<"  B[]:"<<distrB[i]<<std::endl;
            assert(false);
        }
        rel_error_sum+=rel_error;
    }
    std::cout<<text<<"  rel_error_sum:"<<rel_error_sum<<std::endl;
}

void Assortativity::compare_Parts
(
    const DistributedGraph& graph
)
{
    const int my_rank = MPIWrapper::get_my_rank();
    
    auto seq_node_degree = compute_node_degree_distribution_sequential(graph);
    auto par_node_degree = compute_node_degree_distribution(graph);    
    if(my_rank==0)
    {
        compareDistributions(seq_node_degree->first,par_node_degree->first,"node_degree_Out");
        compareDistributions(seq_node_degree->second,par_node_degree->second,"node_degree_In");
        
        auto seq_stdDev = compute_standard_deviation_of_node_degree_distribution_sequential(seq_node_degree);
        auto par_stdDev = compute_standard_deviation_of_node_degree_distribution(par_node_degree);
        
        compareDistributions({seq_stdDev.first},{par_stdDev.first},"stdDev_Out");
        compareDistributions({seq_stdDev.second},{par_stdDev.second},"stdDev_In");
    }
    
    auto seq_normalized_node_degree = compute_normalized_node_degree_distribution_sequential(graph);
    auto par_normalized_node_degree = compute_normalized_node_degree_distribution(graph);    
    if(my_rank==0)
    {
        compareDistributions(seq_normalized_node_degree->first,par_normalized_node_degree->first,"normalized_node_degree_Out");
        compareDistributions(seq_normalized_node_degree->second,par_normalized_node_degree->second,"normalized_node_degree_In");
        
        auto seq_stdDev = compute_standard_deviation_of_node_degree_distribution_sequential(seq_normalized_node_degree);
        auto par_stdDev = compute_standard_deviation_of_node_degree_distribution(par_normalized_node_degree);
        
        compareDistributions({seq_stdDev.first},{par_stdDev.first},"stdDev_Out normalized");
        compareDistributions({seq_stdDev.second},{par_stdDev.second},"stdDev_In normalized");
    }
    
    //auto seq_e_ij = compute_joint_edge_degree_distribution_sequential(graph);
    
    auto par_e_ij = compute_joint_edge_degree_distribution(graph);    
    /*
    if(my_rank==0)
    {
        Distribution2D& seq_e_in_in = std::get<0>(*seq_e_ij);
        Distribution2D& seq_e_in_out = std::get<1>(*seq_e_ij);
        Distribution2D& seq_e_out_in = std::get<2>(*seq_e_ij);
        Distribution2D& seq_e_out_out = std::get<3>(*seq_e_ij);
        
        Distribution2D& par_e_in_in = std::get<0>(*par_e_ij);
        Distribution2D& par_e_in_out = std::get<1>(*par_e_ij);
        Distribution2D& par_e_out_in = std::get<2>(*par_e_ij);
        Distribution2D& par_e_out_out = std::get<3>(*par_e_ij);
        
        compareDistributions(seq_e_in_in.data(),par_e_in_in.data(),"e_in_in");
        compareDistributions(seq_e_in_out.data(),par_e_in_out.data(),"e_in_out");
        compareDistributions(seq_e_out_in.data(),par_e_out_in.data(),"e_out_in");
        compareDistributions(seq_e_out_out.data(),par_e_out_out.data(),"e_out_out");
    }*/
}
