#include "Histogram.h"

std::unique_ptr<Histogram::HistogramData> Histogram::compute_edge_length_histogram_const_bin_width(DistributedGraph& graph, const double bin_width,
												   const unsigned int result_rank) {
	const int number_of_ranks = MPIWrapper::get_number_ranks();
	
	if (result_rank >= number_of_ranks) {
		throw std::invalid_argument("Bad parameter - result_rank" + result_rank);
	}
	if (bin_width <= 0) {
		throw std::invalid_argument("Bad parameter - bin_width");
	}

	std::function<std::unique_ptr<HistogramData>(const double, const double)> bin_width_histogram_creator = [=](const double min_length,
														    const double max_length) {
		double span_length = max_length - min_length;
		/*if (span_length <= 0) {
		    throw std::invalid_argument("Span of edge distribution must be larger than zero!");
		}*/
		unsigned int number_bins = std::ceil(span_length / bin_width);

		    if (span_length == 0) {
			    number_bins = 1;
		    }

		if (number_bins < 1) {
			throw std::invalid_argument("Number of bins must be greater than zero");
		}

		double two_side_overlap_mult = span_length / bin_width - std::floor(span_length / bin_width);
		double one_side_overlap = (two_side_overlap_mult / 2) * bin_width;
		double start_length = min_length - one_side_overlap;

		// Small decrement to avoid comparison errors
		start_length = std::nextafter(start_length, start_length - 1);

		// Small increment to avoid comparison errors
		double bin_width_ext = std::nextafter(bin_width, bin_width + 1);

		auto histogram = std::make_unique<HistogramData>(number_bins);
		for (int i = 0; i < number_bins; i++) {
			(*histogram)[i].first.first = start_length;
			(*histogram)[i].first.second = start_length + bin_width_ext;
			(*histogram)[i].second = 0;
			start_length = start_length + bin_width_ext;
		}

		assert(histogram->front().first.first < min_length);
		assert(histogram->back().first.second > max_length);
		
		return std::move(histogram);
	};

	return std::move(compute_edge_length_histogram(graph, bin_width_histogram_creator, result_rank));
}

std::unique_ptr<Histogram::HistogramData> Histogram::compute_edge_length_histogram_const_bin_count(DistributedGraph& graph, const std::uint64_t bin_count,
												   const unsigned int result_rank) {
	const int number_of_ranks = MPIWrapper::get_number_ranks();
	
	if (result_rank >= number_of_ranks) {
		throw std::invalid_argument("Bad parameter - result_rank" + result_rank);
	}
	if (bin_count < 1) {
		throw std::invalid_argument("Bad parameter - bin_count" + bin_count);
	}

	std::function<std::unique_ptr<HistogramData>(const double, const double)> bin_count_histogram_creator = [=](const double min_length,
														    const double max_length) {
		double span_length = max_length - min_length;
		
		if (span_length < 0) {
			throw std::invalid_argument("Span of edge distribution must not be smaller than zero!");
		}

		    // Solution to create a epsilon that is robust in the case of multiplication and division
		    double epsilon_big = std::numeric_limits<double>::epsilon() * 1000;
		    double epsilon_small = std::numeric_limits<double>::epsilon() * 100;

		double bin_width = (span_length / bin_count) + epsilon_big;

		// Small decrement to avoid comparison errors
		double start_length = min_length - epsilon_small;

		auto histogram = std::make_unique<HistogramData>(bin_count);
		for (int i = 0; i < bin_count; i++) {
			(*histogram)[i].first.first = start_length;
			(*histogram)[i].first.second = start_length + bin_width;
			(*histogram)[i].second = 0;
			start_length = start_length + bin_width;
		}

		assert(histogram->front().first.first < min_length);
		assert(histogram->back().first.second > max_length);
		
		return std::move(histogram);
	};

	return std::move(compute_edge_length_histogram(graph, bin_count_histogram_creator, result_rank));
}

std::unique_ptr<Histogram::HistogramData> Histogram::compute_edge_length_histogram_const_bin_width_sequential(const DistributedGraph& graph, double bin_width,
													      unsigned int result_rank) {
	std::function<std::unique_ptr<HistogramData>(double, double)> bin_width_histogram_creator = [=](double min_length, double max_length) {
		double span_length = max_length - min_length;
		unsigned int number_bins = std::ceil(span_length / bin_width);
		    
		    if (span_length == 0) {
			    number_bins = 1;
		    }

		if (number_bins < 1)
			throw std::invalid_argument("Number of bins must be greater than zero");

		double two_side_overlap_mult = span_length / bin_width - std::floor(span_length / bin_width);
		double one_side_overlap = (two_side_overlap_mult / 2) * bin_width;
		double start_length = min_length - one_side_overlap;

		// Small decrement to avoid comparison errors
		start_length = std::nextafter(start_length, start_length - 1);
		
		// Small increment to avoid comparison errors
		double bin_width_ext = std::nextafter(bin_width, bin_width + 1);

		auto histogram = std::make_unique<HistogramData>(number_bins);
		for (int i = 0; i < number_bins; i++) {
			(*histogram)[i].first.first = start_length;
			(*histogram)[i].first.second = start_length + bin_width;
			(*histogram)[i].second = 0;
			start_length = start_length + bin_width;
		}

		assert(histogram->front().first.first < min_length);
		assert(histogram->back().first.second > max_length);
		
		return std::move(histogram);
	};

	return std::move(compute_edge_length_histogram_sequential(graph, bin_width_histogram_creator, result_rank));
}

std::unique_ptr<Histogram::HistogramData>
Histogram::compute_edge_length_histogram_const_bin_count_sequential(const DistributedGraph& graph, std::uint64_t bin_count, unsigned int result_rank) {
	if (bin_count < 1)
		throw std::invalid_argument("Number of bins must be greater than zero");

	std::function<std::unique_ptr<HistogramData>(double, double)> bin_count_histogram_creator = [=](double min_length, double max_length) {
		double span_length = max_length - min_length;
		
		if (span_length < 0) {
			throw std::invalid_argument("Span of edge distribution must be larger than zero!");
		}
		    
		    // Solution to create a epsilon that is robust in the case of multiplication and division
		    double epsilon_big = std::numeric_limits<double>::epsilon() * 1000;
		    double epsilon_small = std::numeric_limits<double>::epsilon() * 100;

		double bin_width = (span_length / bin_count) + epsilon_big;

		// Small decrement to avoid comparison errors
		double start_length = min_length - epsilon_small;

		auto histogram = std::make_unique<HistogramData>(bin_count);
		for (int i = 0; i < bin_count; i++) {
			(*histogram)[i].first.first = start_length;
			(*histogram)[i].first.second = start_length + bin_width;
			(*histogram)[i].second = 0;
			start_length = start_length + bin_width;
		}
		
		// Small overlap to avoid comparison errors
		histogram->front().first.first = std::nextafter(min_length, min_length - 1);
		histogram->back().first.second = std::nextafter(max_length, max_length + 1);

		assert(histogram->front().first.first < min_length);
		assert(histogram->back().first.second > max_length);
		
		return std::move(histogram);
	};

	return std::move(compute_edge_length_histogram_sequential(graph, bin_count_histogram_creator, result_rank));
}

std::unique_ptr<Histogram::HistogramData>
Histogram::compute_edge_length_histogram(DistributedGraph& graph,
					 const std::function<std::unique_ptr<HistogramData>(const double, const double)> histogram_creator,
					 const unsigned int result_rank) {
	graph.lock_all_rma_windows();

    std::chrono::time_point time = std::chrono::high_resolution_clock::now();
    std::vector<std::chrono::duration<double, std::milli>> times;
    std::vector<std::string> names;

	// Test function parameters
	const int my_rank = MPIWrapper::get_my_rank();
	const int number_ranks = MPIWrapper::get_number_ranks();
	
	if (result_rank >= number_ranks) {
		throw std::invalid_argument("Bad parameter - result_rank:" + result_rank);
	}
	
	const std::uint64_t number_local_nodes = graph.get_number_local_nodes();

	// Compute node local edge lengths of OutEdges
	std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t, std::uint64_t, Vec3d>>>(const DistributedGraph& dg, std::uint64_t node_local_ind)>
	    transfer_node_position = [&](const DistributedGraph& dg, std::uint64_t node_local_ind) {
		    
		    const std::vector<OutEdge>& out_edges = dg.get_out_edges(my_rank, node_local_ind);
		    Vec3d source_node_pos = dg.get_node_position(my_rank, node_local_ind);
		    auto node_position_vec = std::make_unique<std::vector<std::tuple<std::uint64_t, std::uint64_t, Vec3d>>>(out_edges.size());

		    std::transform(out_edges.cbegin(), out_edges.cend(), node_position_vec->begin(), [&](const OutEdge& oEdge) {
			    return std::tuple<std::uint64_t, std::uint64_t, Vec3d>{oEdge.target_rank, oEdge.target_id, source_node_pos};
		    });

		    return std::move(node_position_vec);
	    };

	std::function<double(const DistributedGraph& dg, std::uint64_t node_local_ind, Vec3d para)> compute_edge_length =
	    [&](const DistributedGraph& dg, std::uint64_t node_local_ind, Vec3d source_node_pos) {
		    
		    Vec3d target_node_pos = dg.get_node_position(my_rank, node_local_ind);
		    
		    return (source_node_pos - target_node_pos).calculate_p_norm(2);
	    };

	std::unique_ptr<CommunicationPatterns::NodeToNodeQuestionStructure<Vec3d, double>> edge_length_results =
	    CommunicationPatterns::node_to_node_question<Vec3d, double>(graph, MPIWrapper::MPI_Vec3d, transfer_node_position, MPI_DOUBLE, compute_edge_length);

    times.push_back(std::chrono::high_resolution_clock::now() - time);
    names.push_back("CompLocalEdgeLength");
    time = std::chrono::high_resolution_clock::now();

	// Collect edge lengths of edges to local list
	std::vector<double> edge_lengths;
	for (std::uint64_t node_local_ind = 0; node_local_ind < number_local_nodes; node_local_ind++) {
		
		std::unique_ptr<std::vector<double>> length_of_all_edges = edge_length_results->getAnswersOfQuestionerNode(node_local_ind);
		edge_lengths.insert(edge_lengths.end(), length_of_all_edges->begin(), length_of_all_edges->end());
	}

    times.push_back(std::chrono::high_resolution_clock::now() - time);
    names.push_back("CollectEdgeLengths ");
    time = std::chrono::high_resolution_clock::now();

	// Compute the smallest and largest edge length globally
	double min_len = std::numeric_limits<double>::max();
	double max_len = std::numeric_limits<double>::min();
	
	if (edge_lengths.size() > 0) {
		const auto [min_length, max_length] = std::minmax_element(edge_lengths.begin(), edge_lengths.end());
		min_len = *min_length;
		max_len = *max_length;
	}

	double global_min_length = 0;
	double global_max_length = 0;
	
	MPIWrapper::all_reduce<double>(&min_len, &global_min_length, 1, MPI_DOUBLE, MPI_MIN);
	MPIWrapper::all_reduce<double>(&max_len, &global_max_length, 1, MPI_DOUBLE, MPI_MAX);

    times.push_back(std::chrono::high_resolution_clock::now() - time);
    names.push_back("ComMinMaxLengthEdge");
    time = std::chrono::high_resolution_clock::now();

	// Create histogram with local edge data
	std::unique_ptr<HistogramData> histogram = histogram_creator(global_min_length, global_max_length);
	std::pair<double, double> span = histogram->front().first;
	
	double bin_width = span.second - span.first;
	assert(bin_width > 0);
	double start_length = span.first;
	
	for (const double length : edge_lengths) {
		int index = (length - start_length) / bin_width;
		assert(index < histogram->size());
		assert(index >= 0);
		assert(length >= (*histogram)[index].first.first);
		assert(length < (*histogram)[index].first.second);
		(*histogram)[index].second++;
	}

    times.push_back(std::chrono::high_resolution_clock::now() - time);
    names.push_back("CreateLocHistogram ");
    time = std::chrono::high_resolution_clock::now();

	// Reduce local edge count of histogram to global count
	std::vector<std::uint64_t> histogram_pure_count_src(histogram->size());
	
	for (int i = 0; i < histogram->size(); i++) {
		histogram_pure_count_src[i] = (*histogram)[i].second;
	}

	std::vector<std::uint64_t> histogram_pure_count_dest;
	
	if (my_rank == result_rank) {
		histogram_pure_count_dest.resize(histogram->size());
	}

	MPIWrapper::reduce<std::uint64_t>(histogram_pure_count_src.data(), histogram_pure_count_dest.data(), histogram->size(), MPI_UINT64_T, MPI_SUM,
					  result_rank);

    times.push_back(std::chrono::high_resolution_clock::now() - time);
    names.push_back("CreateGlobHistogram");
    time = std::chrono::high_resolution_clock::now();

	// Reconstruct resulting histogram with global data
	if (my_rank == result_rank) {
		for (int i = 0; i < histogram->size(); i++) {
			(*histogram)[i].second = histogram_pure_count_dest[i];
		}
	} else {
		histogram = std::make_unique<std::vector<std::pair<std::pair<double, double>, std::uint64_t>>>();
	}

    times.push_back(std::chrono::high_resolution_clock::now() - time);
    names.push_back("ReconstrHistogram  ");
    time = std::chrono::high_resolution_clock::now();

	graph.unlock_all_rma_windows();

    std::vector<double> time_double;
    std::for_each(times.cbegin(), times.cend(), [&](auto time) { time_double.push_back(time.count()); });

    std::vector<double> global_avg_times_double(6);
    MPIWrapper::reduce<double>(time_double.data(), global_avg_times_double.data(), 6, MPI_DOUBLE, MPI_SUM, 0);
    std::for_each(global_avg_times_double.begin(), global_avg_times_double.end(),
		  [=](double& time) { time /= number_ranks; });

    std::vector<double> global_max_times_double(6);
    MPIWrapper::reduce<double>(time_double.data(), global_max_times_double.data(), 6, MPI_DOUBLE, MPI_MAX, 0);

    if (my_rank == 0) {
	    std::cout.precision(5);
	    std::cout << "compute_edge_length_histogram" << std::endl;
	    for (int i = 0; i < names.size(); i++) {
		    std::cout << names[i] << ":\tavg:" << global_avg_times_double[i]
			      << "\tmax:" << global_max_times_double[i] << "   milliseconds" << std::endl;
	    }
	    double total_avg = std::accumulate(global_avg_times_double.begin(), global_avg_times_double.end(), 0);
	    double total_max = std::accumulate(global_max_times_double.begin(), global_max_times_double.end(), 0);
	    std::cout << "Total              "
		      << ":\tavg:" << total_avg << "\tmax:" << total_max << "   milliseconds" << std::endl;
	    std::cout << "----------------------------------" << std::endl;
	    fflush(stdout);
    }

	return std::move(histogram);
}

std::unique_ptr<Histogram::HistogramData>
Histogram::compute_edge_length_histogram_sequential(const DistributedGraph& graph,
						    std::function<std::unique_ptr<HistogramData>(double, double)> histogram_creator, unsigned int result_rank) {
	const int my_rank = MPIWrapper::get_my_rank();
	const int number_ranks = MPIWrapper::get_number_ranks();

	// Main rank gathers other ranks number of nodes
	std::vector<std::uint64_t> number_nodes_of_ranks;
	number_nodes_of_ranks.resize(number_ranks);
	std::uint64_t number_local_nodes = graph.get_number_local_nodes();
	
	MPIWrapper::gather<uint64_t>(&number_local_nodes, number_nodes_of_ranks.data(), 1, MPI_UINT64_T, result_rank);
	
	auto histogram = std::make_unique<std::vector<std::pair<std::pair<double, double>, std::uint64_t>>>();

	// Only the main rank collects all edge lengths and sorts each length into the
	// buckets of the histogram created using the supplied function
	if (my_rank == result_rank) {
		std::vector<double> edge_lengths;

		double max_length = 0; // assumption: min edge length is 0
		double min_length = std::numeric_limits<double>::max();

		for (int rank = 0; rank < number_ranks; rank++) {
			for (int node = 0; node < number_nodes_of_ranks[rank]; node++) {
				
				const Vec3d node_position = graph.get_node_position(rank, node);
				const std::vector<OutEdge>& out_edges = graph.get_out_edges(rank, node);

				for (const auto& [target_rank, target_id, weight] : out_edges) {
					const Vec3d target_position = graph.get_node_position(target_rank, target_id);

					const Vec3d difference = target_position - node_position;
					const double length = difference.calculate_2_norm();
					edge_lengths.push_back(length);

					// Find out bounds of the edge lengths for the histogram creation function
					if (length > max_length)
						max_length = length;
					if (length < min_length)
						min_length = length;
				}
			}
		}
		histogram = histogram_creator(min_length, max_length);

		// For each bucket iterate through the edge lengths vector and increase bucket counter if corresponding edge length is found
		for (int i = 0; i < histogram->size(); i++) {
			for (int j = 0; j < edge_lengths.size(); j++) {
				// Count edge lengths inside interval greater equal the lower and smaller than the upper bound
				if (edge_lengths[j] >= (*histogram)[i].first.first &&
				    edge_lengths[j] < (*histogram)[i].first.second) {
					(*histogram)[i].second++;
				}
			}
		}
	}
	
	return std::move(histogram);
}
