#include "DistributedGraph.h"
#include "MPIWrapper.h"

#include "AllPairsShortestPath.h"
#include "DegreeCounter.h"
#include "EdgeCounter.h"
#include "EdgeLength.h"
#include "NodeCounter.h"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include <iostream>
#include <sstream>
#include <string>

int main(int argument_count, char* arguments[]) {
	CLI::App app{ "" };

	std::string input_directory{};
	auto* opt_input_directory = app.add_option("--input", input_directory, "The directory that contains the input files.")->required();
	opt_input_directory->check(CLI::ExistingDirectory);

	CLI11_PARSE(app, argument_count, arguments);

	MPIWrapper::init(argument_count, arguments);

	const auto my_rank = MPIWrapper::get_my_rank();

	{
		std::stringstream ss{};
		ss << "All " << MPIWrapper::get_number_ranks() << " ranks finished loading their local data!" <<'\n';

		DistributedGraph dg(input_directory);

		MPIWrapper::barrier();
		
		if (0 == my_rank) {
			std::cout << ss.str();
			fflush(stdout);
		}

		const auto number_total_nodes = NodeCounter::count_nodes(dg);

		const auto number_total_in_edges = InEdgeCounter::count_in_edges(dg);
		const auto number_total_out_edges = OutEdgeCounter::count_out_edges(dg);

		const auto [min_in, max_in] = InDegreeCounter::count_in_degrees(dg);
		const auto [min_out, max_out] = OutDegreeCounter::count_out_degrees(dg);

		const auto average_edge_length = EdgeLength::compute_edge_length(dg);

		const auto [average_shortest_path, number_unreachables] = AllPairsShortestPath::compute_apsp(dg);

		if (my_rank == 0) {
			std::cout << "The total number of nodes in the graph is: " << number_total_nodes <<'\n';
			std::cout << "The total number of in edges in the graph is: " << number_total_in_edges <<'\n';
			std::cout << "The total number of out edges in the graph is: " << number_total_out_edges <<'\n';
			std::cout << "The minimum in-degree is " << min_in << " and the maximum in-degree is " << max_in <<'\n';
			std::cout << "The minimum out-degree is " << min_out << " and the maximum out-degree is " << max_out << '\n';
			std::cout << "The average edge length is " << average_edge_length << '\n';
			std::cout << "The average shortest path is " << average_shortest_path << ", however, there were " << number_unreachables << " unreachables.\n";
		}
	}

	MPIWrapper::barrier();

	MPIWrapper::finalize();

	return 0;
}
