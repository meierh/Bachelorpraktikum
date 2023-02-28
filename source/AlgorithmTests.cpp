#include "AlgorithmTests.h"
#include "CentralityApprox.cpp" // solved linking problem (maybe needed because file name != class name (?))

void test_algorithm_parallelization
(
	std::filesystem::path input_directory
)
{
	const int my_rank = MPIWrapper::get_my_rank();
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();
	std::string test_result;
	
	
// Test AreaConnectivity algorithm parallelization
	std::unique_ptr<AreaConnectivity::AreaConnecMap> area_connect_parallel;
	std::unique_ptr<AreaConnectivity::AreaConnecMap> area_connect_sequential_helge;
	std::unique_ptr<AreaConnectivity::AreaConnecMap> area_connect_sequential;
	try{
		area_connect_parallel = AreaConnectivity::compute_area_connectivity_strength(dg); // No runtime errors
		MPIWrapper::barrier();
		area_connect_sequential_helge = AreaConnectivity::area_connectivity_strength_sequential_helge(dg); // Runtime errors
		MPIWrapper::barrier();
		area_connect_sequential = AreaConnectivity::area_connectivity_strength_sequential(dg);
		MPIWrapper::barrier();
		compare_area_connec_map(*area_connect_parallel,*area_connect_sequential);
		compare_area_connec_map(*area_connect_parallel,*area_connect_sequential_helge);
		compare_area_connec_map(*area_connect_sequential_helge,*area_connect_sequential);
		
		test_result = "AreaConnectivity test completed";
	}
	catch(std::string error_code)
	{
		test_result = "AreaConnectivity Error :"+error_code;
	}
	if(my_rank==0)
		std::cout<<test_result<<std::endl;
	
// Test Histogram algorithm parallelization
	std::unique_ptr<Histogram::HistogramData> histogram_count_bins;
	std::unique_ptr<Histogram::HistogramData> histogram_width_bins;
	std::unique_ptr<Histogram::HistogramData> histogram_count_bins_sequential;
	std::unique_ptr<Histogram::HistogramData> histogram_width_bins_sequential;
	double bin_width = 1;
	std::uint64_t bin_count = 50;
	try{
		
		histogram_count_bins = Histogram::compute_edge_length_histogram_const_bin_count(dg,bin_count);
		MPIWrapper::barrier();
		histogram_width_bins = Histogram::compute_edge_length_histogram_const_bin_width(dg,bin_width);
		MPIWrapper::barrier();
		histogram_count_bins_sequential = Histogram::compute_edge_length_histogram_const_bin_count_sequential(dg,bin_count);
		MPIWrapper::barrier();
		histogram_width_bins_sequential = Histogram::compute_edge_length_histogram_const_bin_width_sequential(dg,bin_width);
		MPIWrapper::barrier();
		compare_edge_length_histogram(*histogram_count_bins,*histogram_count_bins_sequential,1e-8);
		compare_edge_length_histogram(*histogram_width_bins,*histogram_width_bins_sequential,1e-8);
		
		test_result = "Histogram test completed";
	}
	catch(std::string error_code)
	{
		test_result = "Histogram Error :"+error_code;
	}
	if(my_rank==0)
		std::cout<<test_result<<std::endl;

// Test Modularity algorithm parallelization
	double modularity_par,modularity_seq;
	try{
		modularity_par = Modularity::compute_modularity(dg);
		modularity_seq = Modularity::compute_modularity_sequential(dg);//modularity_par;
		double absolute_error = std::abs(modularity_par-modularity_seq);
		double relative_error = absolute_error / 0.5*(modularity_par+modularity_seq);
		if(relative_error>1e-8)
		{
			std::stringstream error_code;
			error_code<<"modularity_par:"<<modularity_par<<"   modularity_seq:"<<modularity_seq<<"    absolute_error:"<<absolute_error<<"   relative_error:"<<relative_error;
			throw error_code.str();
		}
		
		test_result = "Modularity test completed";
	}
	catch(std::string error_code)
	{
		test_result = "Modularity Error :"+error_code;
	}
	if(my_rank==0)
		std::cout<<test_result<<std::endl;
}

void test_centrality_approx(std::filesystem::path input_directory)
{
	const int my_rank = MPIWrapper::get_my_rank();
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();
	std::string test_result;
	
	
// Test BetweennessCentrality Approximation
	std::unique_ptr<BetweennessCentralityApproximation::BC_e> betweenness_centrality;
	int m = 20;
	double d = 0.25;
	int k = 10;

	std::cout<<"Start betweennessCentralityApprox..."<<std::endl;
	try{
		betweenness_centrality = BetweennessCentralityApproximation::compute_betweenness_centrality_approx(dg, m, d, k);
		MPIWrapper::barrier();
		/*...*/
		test_result = "BetweennessCentralityApprox test completed";
	}
	catch(std::string error_code)
	{
		test_result = "BetweennessCentralityApprox Error :"+error_code;
	}
	if(my_rank==0)
		std::cout<<test_result<<std::endl;
}

void compare_area_connec_map(const AreaConnectivity::AreaConnecMap& map_par,const AreaConnectivity::AreaConnecMap& map_seq)
{
	for(auto key_value=map_par.begin();key_value!=map_par.end();key_value++)
	{
		auto other_key_value = map_seq.find(key_value->first);
		if(other_key_value!=map_seq.end())
		{
			if(other_key_value->second!=key_value->second)
			{
				std::stringstream error_code;
				error_code<<"key_value:"<<key_value->first.first<<" --> "<<key_value->first.second<<"  map_par:"<<key_value->second<<"  map_seq:"<<other_key_value->second;
				throw error_code.str();
			}
		}
		else
		{
			std::stringstream error_code;
			error_code<<"key_value:"<<key_value->first.first<<" --> "<<key_value->first.second<<"  does not exist in map_seq";
			throw error_code.str();
		}
	}
	for(auto key_value=map_seq.begin();key_value!=map_seq.end();key_value++)
	{
		auto other_key_value = map_par.find(key_value->first);
		if(other_key_value!=map_par.end())
		{
			if(other_key_value->second!=key_value->second)
			{
				std::stringstream error_code;
				error_code<<"key_value:"<<key_value->first.first<<" --> "<<key_value->first.second<<"  map_seq:"<<key_value->second<<"  map_par:"<<other_key_value->second;
				throw error_code.str();
			}
		}
		else
		{
			std::stringstream error_code;
			error_code<<"key_value:"<<key_value->first.first<<" --> "<<key_value->first.second<<"  does not exist in map_par";
			throw error_code.str();
		}
	}
	
	if(map_par.size()!=map_seq.size())
	{
		std::stringstream error_code;
		error_code<<"map_par:"<<map_par.size()<<" || "<<"map_seq:"<<map_seq.size();
		throw error_code.str();
	}
}

void compare_edge_length_histogram(const Histogram::HistogramData& histogram_par, const Histogram::HistogramData& histogram_seq, const double epsilon)
{
	const auto my_rank = MPIWrapper::get_my_rank();
	if(my_rank!=0)
		return;

	if(histogram_par.size() != histogram_seq.size())
	{
		std::stringstream error_code;
		error_code<<"histogram_par.size():"<<histogram_par.size()<<"  histogram_par.size()"<<histogram_par.size();
		throw error_code.str();
	}
	
	std::uint64_t total_edges_par = 0;
	for(auto entry: histogram_par)
	{
		total_edges_par += entry.second;
	}
	
	std::uint64_t total_edges_seq = 0;
	for(auto entry: histogram_seq)
	{
		total_edges_seq += entry.second;
	}
	
	if(total_edges_par != total_edges_seq)
	{
		std::stringstream error_code;
		error_code<<"total_edges_par:"<<total_edges_par<<"  total_edges_seq:"<<total_edges_seq;
		throw error_code.str();
	}		

	for(int bin=0; bin<histogram_par.size(); bin++)
	{
		auto elem_par = histogram_par[bin];
		auto elem_seq = histogram_seq[bin];
		if(fabs(elem_par.first.first - elem_seq.first.first) > epsilon)
		{
			std::stringstream error_code;
			error_code<<"Histograms have different bin boundings in bin:"<<bin<<"  elem_par->first.first:"<<elem_par.first.first<<"   elem_seq->first.first:"<<elem_seq.first.first;
			throw error_code.str();
		}
		if(fabs(elem_par.first.second - elem_seq.first.second) > epsilon)
		{
			std::stringstream error_code;
			error_code<<"Histograms have different bin boundings in bin:"<<bin<<"  elem_par->first.second:"<<elem_par.first.second<<"   elem_seq->first.second:"<<elem_seq.first.second;
			throw error_code.str();
		}
		if(elem_par.second != elem_seq.second)
		{
			std::stringstream error_code;
			error_code<<"Histograms have different bin values in bin:"<<bin<<"  elem_par.second:"<<elem_par.second<<"   elem_seq.second:"<<elem_seq.second;
			throw error_code.str();
		}
	}
}
