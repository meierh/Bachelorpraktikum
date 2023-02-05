﻿#include "DistributedGraph.h"
#include "MPIWrapper.h"

#include "AllPairsShortestPath.h"
#include "Centrality.h"
#include "Clustering.h"
#include "DegreeCounter.h"
#include "EdgeCounter.h"
#include "EdgeLength.h"
#include "NodeCounter.h"

#include "GraphProperty.h"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include <csignal>
#include <iostream>
#include <sstream>
#include <string>

void calculate_metrics(std::filesystem::path input_directory) {
	const auto my_rank = MPIWrapper::get_my_rank();

	DistributedGraph dg(input_directory);

	MPIWrapper::barrier();

	if (0 == my_rank) {
		std::cout << "All " << MPIWrapper::get_number_ranks() << " ranks finished loading their local data!" << '\n';
		fflush(stdout);
	}

	MPIWrapper::barrier();

	const auto number_total_nodes = NodeCounter::count_nodes(dg);

	const auto number_total_in_edges = InEdgeCounter::count_in_edges(dg);
	const auto number_total_out_edges = OutEdgeCounter::count_out_edges(dg);

	const auto [min_in, max_in] = InDegreeCounter::count_in_degrees(dg);
	const auto [min_out, max_out] = OutDegreeCounter::count_out_degrees(dg);

	const auto average_edge_length = EdgeLength::compute_edge_length(dg);

	//const auto [average_shortest_path, global_efficiency, number_unreachables] = AllPairsShortestPath::compute_apsp(dg);

	//const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(dg);

	const auto average_cluster_coefficient = Clustering::compuate_average_clustering_coefficient(dg);

	if (my_rank == 0) {
		std::cout << "The total number of nodes in the graph is: " << number_total_nodes << '\n';
		std::cout << "The total number of in edges in the graph is: " << number_total_in_edges << '\n';
		std::cout << "The total number of out edges in the graph is: " << number_total_out_edges << '\n';
		std::cout << "The minimum in-degree is " << min_in << " and the maximum in-degree is " << max_in << '\n';
		std::cout << "The minimum out-degree is " << min_out << " and the maximum out-degree is " << max_out << '\n';
		std::cout << "The average edge length is " << average_edge_length << '\n';
		//std::cout << "The average shortest path is " << average_shortest_path << ", however, " << number_unreachables << " pairs had no path.\n";
		//std::cout << "The global efficiency is " << global_efficiency << ", however, " << number_unreachables << " pairs had no path.\n";
		std::cout << "The average clustering coefficient is: " << average_cluster_coefficient << '\n';
		//std::cout << "The average betweenness centrality is: " << average_betweenness_centrality << '\n';
		fflush(stdout);
	}
}

void testEdgeGetter(std::filesystem::path input_directory) {
	const auto my_rank = MPIWrapper::get_my_rank();

	DistributedGraph dg(input_directory);

	MPIWrapper::barrier();

	if (0 == my_rank) {
		std::cout << "All " << MPIWrapper::get_number_ranks() << " ranks finished loading their local data!" << '\n';
		fflush(stdout);
	}

	int test_rank = 6;
	int test_node = 13;
	
	MPIWrapper::barrier();
	const Vec3d pos_In = dg.get_node_position(test_rank,test_node);
	std::cout << "Rank " << my_rank << " found "<< pos_In <<" in edges in rank "<< test_rank <<'\n';
	MPIWrapper::barrier();
	const auto in = dg.get_number_in_edges(test_rank,test_node);
	std::cout << "Rank " << my_rank << " found "<< in <<" in edges in rank "<< test_rank <<'\n';
	MPIWrapper::barrier();
	const auto out = dg.get_number_out_edges(test_rank,test_node);
	std::cout << "Rank " << my_rank << " found "<< out <<" out edges in rank "<< test_rank <<'\n';
	MPIWrapper::barrier();
	const auto inE = dg.get_in_edge(test_rank,test_node,0);
	std::cout << "Rank " << my_rank << " found "<< inE.source_rank<<" "<<inE.source_id<<" "<<inE.weight <<" in edge in rank "<< test_rank <<'\n';
	MPIWrapper::barrier();
	const auto outE = dg.get_out_edge(test_rank,test_node,2);
	std::cout << "Rank " << my_rank << " found "<< outE.target_rank<<" "<<outE.target_id<<" "<<outE.weight <<" out edge in rank "<< test_rank <<'\n';
	MPIWrapper::barrier();
	
	std::cout<<"Do In"<<std::endl;
	MPIWrapper::barrier();
	try{
		const int number_of_local_nodes = dg.get_number_local_nodes();
		for(int node_local_ind=0;node_local_ind<number_of_local_nodes;node_local_ind++)
		{
			const std::vector<InEdge>& iEdges = dg.get_in_edges(my_rank, node_local_ind);
			if(iEdges.size() != dg.get_number_in_edges(my_rank,node_local_ind))
				throw 1;
			for(InEdge iEdge: iEdges)
			{
				if(iEdge.source_rank>=8 || iEdge.source_rank<0)
					throw 2;
				if(iEdge.source_id>=10000 || iEdge.source_id<0)
					throw 3;
			}
			for(const InEdge& iEdge : iEdges)
			{
				int source_rank = iEdge.source_rank;
				unsigned int source_id = iEdge.source_id;
				int weight = iEdge.weight;
				
				int ident = 0;
				const std::vector<OutEdge>& oEdges = dg.get_out_edges(source_rank,source_id);
				for(const OutEdge& oEdge: oEdges)
				{
					if(oEdge.target_rank>=8 || oEdge.target_rank<0)
						throw 4;
					if(oEdge.target_id>=10000 || oEdge.target_id<0)
						throw 5;
					
					if(my_rank==oEdge.target_rank && node_local_ind==oEdge.target_id && weight==oEdge.weight)
						ident++;
				}
				if(ident!=1)
					throw 6;
			}
		}
		std::cout<<"Done In"<<std::endl;
	}
	catch(int err)
	{
		std::cout<<"In Error:"<<err<<std::endl;
	}

	std::cout<<"Do Out"<<std::endl;
	MPIWrapper::barrier();
	try{
		const int number_of_local_nodes = dg.get_number_local_nodes();
		for(int node_local_ind=0;node_local_ind<number_of_local_nodes;node_local_ind++)
		{
			const std::vector<OutEdge>& oEdges = dg.get_out_edges(my_rank, node_local_ind);
			if(oEdges.size() != dg.get_number_out_edges(my_rank,node_local_ind))
				throw 1;
			for(OutEdge oEdge: oEdges)
			{
				if(oEdge.target_rank>=8 || oEdge.target_rank<0)
					throw 2;
				if(oEdge.target_id>=10000 || oEdge.target_id<0)
					throw 3;
			}
			for(const OutEdge& oEdge : oEdges)
			{
				int target_rank = oEdge.target_rank;
				unsigned int target_id = oEdge.target_id;
				int weight = oEdge.weight;
				
				int ident = 0;
				const std::vector<InEdge>& iEdges = dg.get_in_edges(target_rank,target_id);
				for(const InEdge& iEdge: iEdges)
				{
					if(iEdge.source_rank>=8 || iEdge.source_rank<0)
						throw 4;
					if(iEdge.source_id>=10000 || iEdge.source_id<0)
						throw 5;
					
					if(my_rank==iEdge.source_rank && node_local_ind==iEdge.source_id && weight==iEdge.weight)
						ident++;
				}
				if(ident!=1)
					throw 6;
			}
			
		}
		std::cout<<"Done Out"<<std::endl;
		}
	catch(int err)
	{
		std::cout<<"Out Error:"<<err<<std::endl;
	}
}

void compareAreaConnecMap(const GraphProperty::AreaConnecMap& mapPar,const GraphProperty::AreaConnecMap& mapSeq)
{
	for(auto keyValue=mapPar.begin();keyValue!=mapPar.end();keyValue++)
	{
		auto otherKeyValue = mapSeq.find(keyValue->first);
		if(otherKeyValue!=mapSeq.end())
		{
			if(otherKeyValue->second!=keyValue->second)
			{
				std::cout<<"keyValue:"<<keyValue->first.first<<" --> "<<keyValue->first.second<<"  mapPar:"<<keyValue->second<<"  mapSeq:"<<otherKeyValue->second<<std::endl;
				throw "Error found";
			}
		}
		else
		{
			std::cout<<"keyValue:"<<keyValue->first.first<<" --> "<<keyValue->first.second<<"  does not exist in mapSeq"<<std::endl;
			throw "Error found";
		}
	}
	for(auto keyValue=mapSeq.begin();keyValue!=mapSeq.end();keyValue++)
	{
		auto otherKeyValue = mapPar.find(keyValue->first);
		if(otherKeyValue!=mapPar.end())
		{
			if(otherKeyValue->second!=keyValue->second)
			{
				std::cout<<"keyValue:"<<keyValue->first.first<<" --> "<<keyValue->first.second<<"  mapSeq:"<<keyValue->second<<"  mapPar:"<<otherKeyValue->second<<std::endl;
				throw "Error found";
			}
		}
		else
		{
			std::cout<<"keyValue:"<<keyValue->first.first<<" --> "<<keyValue->first.second<<"  does not exist in mapPar"<<std::endl;
			throw "Error found";
		}
	}
	
	std::cout<<"mapPar:"<<mapPar.size()<<" || "<<"mapSeq:"<<mapSeq.size()<<std::endl;
}

bool compareEdgeLengthHistogram(const GraphProperty::Histogram& histogramPar, const GraphProperty::Histogram& histogramSeq, const double epsilon)
{
	const auto my_rank = MPIWrapper::get_my_rank();
	if(my_rank!=0)
		return true;
	std::cout << "----------"<<my_rank<<" Compare histograms (epsilon = " << epsilon << "): ----------" << std::endl;
	std::cout << "size of histogramPar:" << histogramPar.size() << std::endl;
	std::cout << "size of histogramSeq:" << histogramSeq.size() << std::endl;

	if(histogramPar.size() != histogramSeq.size())
	{
		throw "Error found: Histograms have different sizes";
	}
	
	std::uint64_t total_Edges_Par = 0;
	for(auto entry: histogramPar)
	{
		total_Edges_Par += entry.second;
	}
	
	std::uint64_t total_Edges_Seq = 0;
	for(auto entry: histogramSeq)
	{
		total_Edges_Seq += entry.second;
	}
	
	if(total_Edges_Par != total_Edges_Seq)
	{
		std::cout<<"total_Edges_Par:"<<total_Edges_Par<<std::endl;
		std::cout<<"total_Edges_Seq:"<<total_Edges_Seq<<std::endl;
		throw "Error found: Histograms have different total size";
	}
	std::cout<<"Equal edge count"<<std::endl;
		

	for(int bin=0; bin<histogramPar.size(); bin++)
	{
		auto elemPar = histogramPar[bin];
		auto elemSeq = histogramSeq[bin];
		if(fabs(elemPar.first.first - elemSeq.first.first) > epsilon)
		{
			std::cout<<"elemPar->first.first:"<<elemPar.first.first<<std::endl;
			std::cout<<"elemSeq->first.first:"<<elemSeq.first.first<<std::endl;
			throw "Error found: Histograms have different bin boundings";
		}
		if(fabs(elemPar.first.second - elemSeq.first.second) > epsilon)
		{
			std::cout<<"elemPar->first.second:"<<elemPar.first.second<<std::endl;
			std::cout<<"elemSeq->first.second:"<<elemSeq.first.second<<std::endl;
			throw "Error found: Histograms have different bin boundings";
		}
		if(elemPar.second != elemSeq.second)
		{
			std::cout<<"---------------------"<<bin<<"--------------------"<<std::endl;
			std::cout<<"elemPar.second:"<<elemPar.second<<std::endl;
			std::cout<<"elemSeq.second:"<<elemSeq.second<<std::endl;
			throw "Error found: Histograms have different counts";
		}
	}
	std::cout << "===> Histograms are equal" << std::endl;
	return true;
}

void test_areaConnectivityStrength(std::filesystem::path input_directory)
{
	const auto my_rank = MPIWrapper::get_my_rank();
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();

	std::unique_ptr<GraphProperty::AreaConnecMap> areaConnectParallel;
	std::unique_ptr<GraphProperty::AreaConnecMap> areaConnectSingleProc_Helge;
	std::unique_ptr<GraphProperty::AreaConnecMap> areaConnectSingleProc;
	try{
		areaConnectParallel = GraphProperty::areaConnectivityStrength(dg); // No runtime errors
		MPIWrapper::barrier();
		areaConnectSingleProc_Helge = GraphProperty::areaConnectivityStrengthSingleProc_Helge(dg); // Runtime errors
		MPIWrapper::barrier();
		areaConnectSingleProc = GraphProperty::areaConnectivityStrengthSingleProc(dg);
		fflush(stdout);
		MPIWrapper::barrier();
		fflush(stdout);
		std::cout<<"------------------------------------------------------------------------------------------"<<std::endl;
		compareAreaConnecMap(*areaConnectParallel,*areaConnectSingleProc);
		compareAreaConnecMap(*areaConnectParallel,*areaConnectSingleProc_Helge);
		compareAreaConnecMap(*areaConnectSingleProc_Helge,*areaConnectSingleProc);
	}
	catch(std::string err)
	{
		std::cout<<"Err:"<<err<<std::endl;
	}
}

void test_histogram(std::filesystem::path input_directory)
{
	const auto my_rank = MPIWrapper::get_my_rank();
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();

	std::unique_ptr<GraphProperty::Histogram> histogramCountBins;
	std::unique_ptr<GraphProperty::Histogram> histogramWidthBins;
	std::unique_ptr<GraphProperty::Histogram> histogramCountBins_SingleProc;
	std::unique_ptr<GraphProperty::Histogram> histogramWidthBins_SingleProc;
	double bin_width = 1;
	std::uint64_t bin_count = 50;
	try{
		
		histogramCountBins = GraphProperty::edgeLengthHistogramm_constBinCount(dg,bin_count);
		MPIWrapper::barrier();
		histogramWidthBins = GraphProperty::edgeLengthHistogramm_constBinWidth(dg,bin_width); // No runtime errors
		MPIWrapper::barrier();
		histogramCountBins_SingleProc = GraphProperty::edgeLengthHistogramm_constBinCountSingleProc(dg,bin_count); // No runtime errors
		MPIWrapper::barrier();
		histogramWidthBins_SingleProc = GraphProperty::edgeLengthHistogramm_constBinWidthSingleProc(dg,bin_width); // No runtime errors
		MPIWrapper::barrier();
		
		fflush(stdout);
		MPIWrapper::barrier();
		fflush(stdout);
		std::cout<<"------------------------------------------------------------------------------------------"<<std::endl;
		compareEdgeLengthHistogram(*histogramCountBins,*histogramCountBins_SingleProc,0.00000001);
		compareEdgeLengthHistogram(*histogramWidthBins,*histogramWidthBins_SingleProc,0.00000001);
	}
	catch(std::string err)
	{
		std::cout<<"Err:"<<err<<std::endl;
	}
}

void test_modularity(std::filesystem::path input_directory)
{
	const auto my_rank = MPIWrapper::get_my_rank();
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();
	
	double modularityPar,modularitySeq;

	try{
		modularityPar = GraphProperty::computeModularity(dg);
		modularitySeq = GraphProperty::computeModularitySingleProc(dg);
		if(my_rank==0)
			std::cout<<"modularityPar:"<<modularityPar<<" modularitySeq:"<<modularitySeq<<std::endl;
	}
	catch(std::string err)
	{
		std::cout<<"Err:"<<err<<std::endl;
	}
}

void test_GraphPropertyAlgorithms(std::filesystem::path input_directory)
{
	const auto my_rank = MPIWrapper::get_my_rank();
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();

	std::unique_ptr<GraphProperty::AreaConnecMap> areaConnect;
	std::unique_ptr<GraphProperty::Histogram> histogramCountBins;
	std::unique_ptr<GraphProperty::Histogram> histogramWidthBins;
	std::vector<long double> motifFraction;
	double modularity;
	try{
		areaConnect = GraphProperty::areaConnectivityStrength(dg);
		//histogramCountBins = GraphProperty::edgeLengthHistogramm_constBinCount(dg,50);
		//histogramWidthBins =  GraphProperty::edgeLengthHistogramm_constBinWidth(dg,2.0);
		//modularity = GraphProperty::computeModularity(dg);
		if (my_rank == 0) 
		{
			//std::cout << "Modularity"<< '\n';
			//std::cout <<modularity<<"   ";
			//fflush(stdout);
		}
	}
	catch(std::string err)
	{
		std::cout<<"Err:"<<err<<std::endl;
	}
}

void test_GraphPropertyAlgorithmsSingleProc(std::filesystem::path input_directory)
{
	const auto my_rank = MPIWrapper::get_my_rank();
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();

	//std::unique_ptr<GraphProperty::AreaConnecMap> areaConnect;
	std::unique_ptr<GraphProperty::Histogram> histogramCountBins;
	std::unique_ptr<GraphProperty::Histogram> histogramWidthBins;
	double modularityPar;
	double modularitySer;
	try{
		/*if(my_rank == 0) std::cout << "areaConnectivityStrengthSingleProc: " << std::endl;
		auto areaConnect = GraphProperty::areaConnectivityStrengthSingleProc(dg);*/
		
		/*
		if(my_rank == 0) std::cout << "edgeLengthHistogram_constBinCount: " << std::endl;
		histogramCountBins = GraphProperty::edgeLengthHistogramm_constBinCount(dg,50);
		MPIWrapper::barrier();
		if(my_rank == 0) compareEdgeLengthHistogram(*histogramCountBins, *histogramCountBins, 0.5);	//nonsensical test :)
		MPIWrapper::barrier();

		if(my_rank == 0) std::cout << "edgeLengthHistogram_constBinWidth: " << std::endl;
		histogramWidthBins = GraphProperty::edgeLengthHistogramm_constBinWidth(dg,2.0);
		MPIWrapper::barrier();
		if(my_rank == 0) compareEdgeLengthHistogram(*histogramWidthBins, *histogramWidthBins, 0.5); //nonsensical test :)
		MPIWrapper::barrier();
		*/

		if(my_rank == 0) std::cout << "computeModularitySingleProc: " << std::endl;
		modularitySer =  GraphProperty::computeModularitySingleProc(dg);
		MPIWrapper::barrier();

		if(my_rank == 0) std::cout << "computeModularity: " << std::endl;
		modularityPar =  GraphProperty::computeModularity(dg);
		if(my_rank == 0) std::cout << "Modularity (parallel): " << modularityPar << std::endl;
		MPIWrapper::barrier();
	}
	catch(std::string err)
	{
		std::cout<<"Err:"<<err<<std::endl;
	}
}


int main(int argument_count, char* arguments[]) {
	CLI::App app{ "" };

	std::string input_directory{};
	auto* opt_input_directory = app.add_option("--input", input_directory, "The directory that contains the input files.")->required();
	opt_input_directory->check(CLI::ExistingDirectory);

	CLI11_PARSE(app, argument_count, arguments);

	MPIWrapper::init(argument_count, arguments);

	test_modularity(input_directory);
	//test_GraphPropertyAlgorithmsSingleProc(input_directory);

	MPIWrapper::finalize();

	return 0;
}
