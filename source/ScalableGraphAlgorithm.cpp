﻿#include "DistributedGraph.h"
#include "MPIWrapper.h"

#include "AllPairsShortestPath.h"
#include "Centrality.h"
#include "Clustering.h"
#include "DegreeCounter.h"
#include "EdgeCounter.h"
#include "EdgeLength.h"
#include "NodeCounter.h"

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

	//const auto average_cluster_coefficient = Clustering::compuate_average_clustering_coefficient(dg);
	
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
	
	MPIWrapper::barrier();

	if (my_rank == 0) {
		std::cout << "The total number of nodes in the graph is: " << number_total_nodes << '\n';
		std::cout << "The total number of in edges in the graph is: " << number_total_in_edges << '\n';
		std::cout << "The total number of out edges in the graph is: " << number_total_out_edges << '\n';
		std::cout << "The minimum in-degree is " << min_in << " and the maximum in-degree is " << max_in << '\n';
		std::cout << "The minimum out-degree is " << min_out << " and the maximum out-degree is " << max_out << '\n';
		std::cout << "The average edge length is " << average_edge_length << '\n';
		//std::cout << "The average shortest path is " << average_shortest_path << ", however, " << number_unreachables << " pairs had no path.\n";
		//std::cout << "The global efficiency is " << global_efficiency << ", however, " << number_unreachables << " pairs had no path.\n";
		//std::cout << "The average clustering coefficient is: " << average_cluster_coefficient << '\n';
		//std::cout << "The average betweenness centrality is: " << average_betweenness_centrality << '\n';
		fflush(stdout);
	}
}

int main(int argument_count, char* arguments[]) {
	CLI::App app{ "" };

	std::string input_directory{};
	auto* opt_input_directory = app.add_option("--input", input_directory, "The directory that contains the input files.")->required();
	opt_input_directory->check(CLI::ExistingDirectory);

	CLI11_PARSE(app, argument_count, arguments);

	MPIWrapper::init(argument_count, arguments);

	calculate_metrics(input_directory);

	MPIWrapper::finalize();

	return 0;
}
