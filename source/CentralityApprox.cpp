#include "CentralityApprox.h"

/*std::vector<std::vector<NodePath>> compute_sssp(const DistributedGraph& graph, unsigned int node_id, unsigned int dest_id, std::uint64_t total_number_nodes, const std::vector<std::uint64_t>& prefix_distribution);
std::unordered_map<int, double> getFunctionValues(std::pair<int, int> sample);
std::pair<int, int> drawSample(const DistributedGraph& graph, int number_ranks, int number_local_nodes, std::vector<std::uint64_t> prefix_distribution);
std::vector<int> drawRademacher(int k);*/

/*
This Approximation is based on the Bavarian Framework with ABRA estimator.
*/
std::unique_ptr<BetweennessCentralityApproximation::BC_e> BetweennessCentralityApproximation::compute_betweenness_centrality_approx(const DistributedGraph& graph, int m, double d, int k, unsigned int result_rank) {
    const int my_rank = MPIWrapper::get_my_rank();
    const int number_ranks = MPIWrapper::get_number_ranks();
    const auto number_local_nodes = graph.get_number_local_nodes();
    const auto total_number_nodes = NodeCounter::all_count_nodes(graph);
    const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(graph);
	const auto prefix_distribution = calculate_prefix_sum(node_distribution);
    std::vector<uint64_t> node_numbers = MPIWrapper::all_gather(graph.get_number_local_nodes());
    
    if(m <= 0){
        std::cout << "Error: m must be greater than 0" << std::endl;
    }

    std::vector<std::vector<double>> sums;   //std::unordered_map<int, std::vector<double>> sums;

    std::vector<double> vec(k+2, std::numeric_limits<double>::infinity());
    vec[k] = vec[k+1] = 0;

    // insert all local nodes into sums (unique ids)
    for(int global_node = 0; global_node < total_number_nodes; global_node++){
        sums.push_back(vec);
    }
    /*for (int node_id_rank = 0; node_id_rank < number_local_nodes; node_id_rank++) {
        const auto node_id = prefix_distribution[my_rank] + node_id_rank;
        sums.insert({node_id, vec});
    }*/

    int iterations = std::floor(m / number_ranks);

    if (my_rank < (m % number_ranks)) {
        iterations++;
    }
    
    // main loop
    for (int i = 0; i < iterations; i++) {
        std::pair<int, int> sample = drawSample(graph, number_ranks, number_local_nodes, prefix_distribution);
        std::unordered_map<int, double> Z = getFunctionValues(graph, sample, prefix_distribution);
        std::vector<double> lambda = drawRademacher(k);

        for (std::pair<int, double> z : Z) {
            int node_id = z.first;
            double function_val = z.second;

            std::vector<double> lambda_modified = lambda;
            lambda_modified.resize(k+2, 1);
            lambda_modified[k] = function_val;

            for (int j = 0; j < k+2; j++) {
                lambda_modified[j] = sums.at(node_id)[j] + function_val * lambda_modified[j];
            }

            sums[node_id] = lambda_modified; //[]-operator updats value or inserts new entry
            //sums.emplace(node_id, lambda_modified);
        }
    }
    
    // Gather sums vectors from all ranks to result rank
    std::function<std::unique_ptr<std::vector<std::pair<std::vector<double>, int>>>()> 
        get_names = [&]() {
            auto data = std::make_unique<std::vector<std::pair<std::vector<double>, int>>>(sums.size());
            std::transform(sums.cbegin(), sums.cend(), data->begin(), [] (std::vector<double> vec) {return std::pair<std::vector<double>, int>(vec, vec.size());});            
            return std::move(data);
        };

    std::unique_ptr<std::vector<std::vector<std::vector<double>>>> sums_of_ranks = gather_Data_to_one_Rank<std::vector<double>, double>
    (
    	graph,
    	get_vectors,
    	[] (std::vector<double> vec_of_doubles) {return std::vector<double>(vec_of_doubles.cbegin(), vec_of_doubles.cend());},
    	[] (std::vector<double> vec_of_doubles) {return std::vector<double>(vec_of_doubles.cbegin(), vec_of_doubles.cend());},
    	MPI_DOUBLE,
    	result_rank
    );
    
    // Result rank adds sums vectors of the other ranks to his sums vector
    if(my_rank == result_rank) {
        
        for(int rank = 0; rank < number_ranks; rank++){
            if(rank == my_rank) 
                continue;

            for (int global_node = 0; global_node < total_number_nodes; global_node++){
                for (int i = 0; i < k+2; i++) {
                    sums[global_node][i] += (*sums_of_ranks).at(rank)[global_node][i];
                } 
            }
        }
    }
    
    // Result vector computes B
    std::unordered_map<std::pair<int, int>, double, BC_hash> bc;
    
    if(my_rank == result_rank){
        
        int global_node = 0;
        for(int rank = 0; rank < number_ranks; rank++){
            for(int local_node = 0; local_node < node_numbers[rank]; local_node++){
                std::pair<int, int> node(rank, local_node);
                bc.insert(std::pair<std::pair<int, int>, double>(node, sums.at(global_node)[k+1]/m));
                global_node++;
            }
        }
    }
    /*
    for (int global_node = 0; global_node < sums.size(); global_node++) {
        int local_node = global_node - prefix_distribution[my_rank];
        Node node = {my_rank, local_node};
        B.insert({node, sums.at(global_node)[k+1]/m});
    }
    */
    /*for (std::pair<int, vector<double>> bc_pair : sums) {
        int node_id_rank = bc_pair.first - prefix_distribution[my_rank];
        node n = {my_rank, node_id_rank};
        B.insert({n, bc_pair.second[k+1]/m});
    }*/

    ///////////////////// Start: Merge sums to one rank: /////////////////////
    
    /* Getherv Implementation aus MPIWrapper im feature_networkMotifs branch verwendet:
    template<typename T>
	static void gatherv(T* src, int count, T* dest, int* destCounts, int* displs, MPI_Datatype datatype,int root){
		if (const auto error_code= MPI_Gatherv(src,count,datatype,dest,destCounts,displs,datatype,root,MPI_COMM_WORLD);	error_code != 0) {
			std::cout << "Gatherving all values returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}
    */
    /*
    // Init variables for gatherv:
    std::vector<int> sums_values;               // send buffer
    int number_sums_values;                     // send buffer size
    std::vector<int> all_sums_values;           // receive buffer
    std::vector<int> rank_to_number_sums_values;// receive counts
    std::vector<int> sums_values_displ;         // displacement

    // Following prepare variables for gatherv:
    
    // Flatten all value vectors in the map to one-dimensional sums_values vector
    for (auto const& pair : sums){
        for(auto const& elem : pair.second){
            sums_values.push_back(elem);
            number_sums_values++;
        }
    }
    assert(number_sums_values == (number_local_nodes * (k+2)));   //debug
    
    // Gather from each rank number_sums_values for the gatherv receive counts
    if(my_rank == result_rank){
        rank_to_number_sums_values.resize(number_ranks);
    }
    MPIWrapper::gather<int>(&number_sums_values, rank_to_number_sums_values.data(), 1, MPI_INT, result_rank);
    MPIWrapper::barrier();

    // Displacement contains the offset indexes for each rank so that gatherv knows where the insertion of local buffer elements start in the receive buffer
    if(my_rank == result_rank){
        sums_values_displ.resize(number_ranks);
        int displacement = 0;
        
        for(size_t r = 0; r < number_ranks; r++){
            sums_values_displ[r] = displacement;
            displacement += rank_to_number_sums_values[r];
        }
        all_sums_values.resize(displacement);
    }

    // Wrapper::gatherv(send_buffer, send_buffer_size, receive_buffer, receive_counts, displacement, mpi_data_type, result_rank)
    MPIWrapper::gatherv<int>(sums_values.data(), number_sums_values,
                            all_sums_values.data(), rank_to_number_sums_values.data(), 
                            sums_values_displ.data(), MPI_INT, result_rank);
    MPIWrapper::barrier();

    // Group elements of all_sums_values to value_vectors and sort them into the gathered sums map
    std::unordered_map<int, std::vector<double>> sums_gathered;
    if(my_rank == result_rank){
        for (size_t i = 0; i < total_number_nodes; i++){   
            std::vector<int> value_vector;
            int key = i;
            for (size_t j = i; j < k+2; j++){
                value_vector.push_back(all_sums_values[j]);
            }
            sums_gathered.emplace(key, value_vector);
        }   
    }
    */
    ///////////////////// End: Merge sums to one rank: /////////////////////
    
    // result_rank computes epsilon
    if (my_rank == result_rank) {
        
        double epsilon = getEpsilon(sums, m, d, k);
        return std::make_unique<BC_e>(std::pair<std::unordered_map<std::pair<int, int>, double, BC_hash>, double>(bc, epsilon));
    }

    return std::make_unique<BC_e>();
}


    /*
        Compute shortest paths
        Returns: shortest paths from given startnode to the given destinationnode
    */
std::vector<std::vector<NodePath>> BetweennessCentralityApproximation::compute_sssp(const DistributedGraph& graph, unsigned int node_id, unsigned int dest_id, std::uint64_t total_number_nodes, const std::vector<std::uint64_t>& prefix_distribution) {
        const auto my_rank = MPIWrapper::get_my_rank();

        // initialize all the necessary datastructeres
		std::vector<std::vector<NodePath>> shortest_paths(total_number_nodes, std::vector<NodePath>{});
		std::vector<double> distances(total_number_nodes, std::numeric_limits<double>::infinity());

		std::priority_queue<VertexDistancePath, std::vector<VertexDistancePath>, std::greater<VertexDistancePath>> shortest_paths_queue{};

		const auto root_id = prefix_distribution[my_rank] + node_id;

		distances[root_id] = 0;

		NodePath start(20);
		start.append_node(node_id);

		shortest_paths[root_id] = { start };
		shortest_paths_queue.emplace(my_rank, node_id, 0, std::move(start));

		while (!shortest_paths_queue.empty()) {
			const auto [current_rank, current_id, current_distance, current_path] = shortest_paths_queue.top();
			shortest_paths_queue.pop();

			const auto& out_edges = graph.get_out_edges(current_rank, current_id);

            // Updating all nodes which are the head from the outgoing edges of the current node
			for (const auto& [target_rank, target_id, weight] : out_edges) {
				const auto new_distance = current_distance + std::abs(weight);

				const auto other_node_id = prefix_distribution[target_rank] + target_id;

                // No update 
				if (distances[other_node_id] < new_distance) {
					continue;
				}

                // Update: update distance and clear the shortest path to this node ()
				if (distances[other_node_id] > new_distance) {
					distances[other_node_id] = new_distance;
					shortest_paths[other_node_id].clear();
				}

				NodePath new_path = current_path;
				new_path.append_node(other_node_id);

				shortest_paths_queue.emplace(target_rank, target_id, new_distance, new_path);
				shortest_paths[other_node_id].emplace_back(std::move(new_path));
			}

            // predicate for checking if we reached the destination
            if(current_id == dest_id) {
                break;
            }
		}

        return shortest_paths;
    }

/*
    Update Function (Function getInternal..., add to map, correct syntax)
*/
std::unordered_map<int, double> BetweennessCentralityApproximation::getFunctionValues(const DistributedGraph& graph, std::pair<int, int> sample, const std::vector<std::uint64_t>& prefix_distribution) {

    /*
    
        MAYBE CHANAGE PARAMETERS TO UINT_64
    
    */

    std::unordered_map<int, double> z{};
    const auto total_number_nodes = NodeCounter::all_count_nodes(graph);
    std::vector<std::vector<NodePath>> all_shortest_paths_to_all_nodes = compute_sssp(graph, sample.first, sample.second, total_number_nodes, prefix_distribution);
    int uv = all_shortest_paths_to_all_nodes[sample.second].size();
    std::vector<NodePath> shortest_paths = all_shortest_paths_to_all_nodes[sample.second];
    
    
    /*
        Iterate over the shortest_paths[sample.t2]
        increment if internal node was found or insert it for the first time in a temporary map
    */
    std::unordered_map<int, double> temp_map{};
    for(int i = 0; i < shortest_paths.size(); i++) {
        std::vector<std::uint64_t> nodes_on_path = shortest_paths[i].get_nodes();
        // Starts at index one and ends on before the last index of path -> first and last node of a path should not count in SP for BC
        for(int j = 1; j < nodes_on_path.size()-1; j++) {   //(nodes.size()-1 zu nodes_on_path.size()-1 geändert)
            // Map contains internal node -> increment
            if(temp_map.count(nodes_on_path[j]) != 0) {
                temp_map[nodes_on_path[j]]++;
            }
            // add internal node for the first time
            else {
                temp_map.insert({nodes_on_path[j], 1});
            }
        }
    }
    
    int uvw = 0;
    for(int i = 0; i < temp_map.size(); i++) {
        // get value from temporary map
        // int uvw = ;
        double value = uvw/(double)temp_map[i];
        if(value != 0) {
            z.insert({i, value});
        }
    }

    return z;
}    
/*
This function is called by each rank. It draws a pair of distinct nodes uniformly among all possible pairs from the given graph.
*/
std::pair<int, int> BetweennessCentralityApproximation::drawSample(const DistributedGraph& graph, int number_ranks, int number_local_nodes, std::vector<std::uint64_t> prefix_distribution) {
    std::random_device dev; // set up uniform distribution
    std::mt19937 gen(dev()); // PRNG
    std::uniform_int_distribution<int> rank_dist(0, number_ranks);
    std::uniform_int_distribution<int> node_dist(0, number_local_nodes);

    int node1 = prefix_distribution[rank_dist(gen)] + node_dist(gen);
    int node2 = prefix_distribution[rank_dist(gen)] + node_dist(gen);

    while (node1 == node2) {
        node2 = prefix_distribution[rank_dist(gen)] + node_dist(gen);
    }
    
    return {node1, node2};
}


/*
Creates a vector of size k with elements that are 1 or -1, uniformly drawn.
*/
std::vector<double> BetweennessCentralityApproximation::drawRademacher(int k) {
    std::random_device dev; // set up uniform distribution
    std::mt19937 gen(dev()); // PRNG
    std::uniform_int_distribution<int> dist(0, 1);

    std::vector<double> lambda(k, 1.0);

    for (int i = 0; i < k; i++) {
        if (! dist(gen)) lambda[i] = -1.0;
    }

    return lambda;
}


/*
Computes probabilistically-guaranteed accuracy epsilon.
*/
double BetweennessCentralityApproximation::getEpsilon(std::vector<std::vector<double>> sums, int m, double d, int k) {
    double eta = d; // implementation might change
    double log_val = log((double)5/eta);

    // compute Rademacher Average R and beta
    double R = 0.0;
    for (int z = 0; z < k; z++) {
        double max;
        for (int i = 0; i < sums.size(); i++) { //for (auto iter = sums.begin(); iter != sums.end(); iter++) {
            //std::vector<double> vec = iter->second;
            //if (vec[k] > beta) beta = vec[k];
            if (sums.at(i)[z] > max) max = sums.at(i)[z];
        }
        R += max / (double) m;
    }
    R = R / (double) k;

    // compute beta
    double beta = 0.0;
    for (int i = 0; i < sums.size(); i++) { //for (auto iter = sums.begin(); iter != sums.end(); iter++) {
        //std::vector<double> vec = iter->second;
        //if (vec[k] > beta) beta = vec[k];
        if (sums.at(i)[k] > beta) beta = sums.at(i)[k];
    }

    // compute gamma
    double gamma = beta + (2*log_val) / (double) (3*m) + 
                    std::sqrt( (log_val / (m*std::sqrt(3))) * (log_val / (m*std::sqrt(3))) + (2*beta*log_val / (double) m) );

    // compute rho
    double rho = R + (2*log_val) / (3*k*m) + std::sqrt( (4*beta*log_val) / (double) (k*m) );

    // compute r
    double r = rho + (log_val) / (double) (3*m) + 
                    std::sqrt( (log_val / (2*m*std::sqrt(3))) * (log_val / (2*m*std::sqrt(3))) + (rho*log_val / (double) m) );

    // compute epsilon
    double epsilon = 2*r + log_val / (double) (3*m) + std::sqrt(2*(gamma + 4*r)*log_val / (double) m);

    return epsilon;
}