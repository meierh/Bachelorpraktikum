#pragma once

#include "CommunicationPatterns.h"
#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include <cassert>
#include <functional>
#include <stdexcept>
#include <unordered_map>

class AreaConnectivity {
public:
	/*|||-----------compute_area_connectivity_strength---------------------
	 *
	 * Foreach combination of areas A and B, the function sums the weight
	 * of all edges connecting a node in area A with a node in area B.
	 *
	 * Parameters
	 * graph:           A DistributedGraph (Function is MPI compliant)
	 * result_rank:    MPI Rank to receive the results
	 *
	 * Returns: std::unordered_map with the a pair of area names as key
	 *          and the summed weight as value.
	 *          (std::pair(area_name_A,area_name_B)->summed_weight)
	 * 
	 * MPI Constraints: 	Function must be called by all ranks simultaneously
	 * 						Function returns correct data only to rank stated in
	 * 						the parameter result_rank
	 */
	struct StdPair_hash {
		template <class T1, class T2>
		std::size_t operator()(const std::pair<T1, T2>& p) const {
			return std::hash<T1>{}(p.first) ^ std::hash<T2>{}(p.second);
		}
	};
	using AreaConnecMap = std::unordered_map<std::pair<std::string, std::string>, int, StdPair_hash>;
	static std::unique_ptr<AreaConnecMap> compute_area_connectivity_strength(DistributedGraph& graph,
										 unsigned int result_rank = 0);
	/*|||-----------compute_area_connectivity_strength---------------------*/

	static std::unique_ptr<AreaConnecMap> area_connectivity_strength_sequential(const DistributedGraph& graph,
										    unsigned int result_rank = 0);

private:
	using AreaLocalID = std::pair<std::uint64_t, std::uint64_t>;
	
	struct StdDoublePair_hash {
		StdPair_hash hash;
		template <class T1, class T2>
		std::size_t operator()(const std::pair<T1, T2>& p) const {
			return hash(p.first) ^ hash(p.second);
		}
	};

	using AreaIDConnecMap = std::unordered_map<std::pair<AreaLocalID, AreaLocalID>, int, StdDoublePair_hash>;
	
	typedef struct {
		std::int64_t source_rank;
		std::int64_t source_area_localID;
		std::int64_t target_rank;
		std::int64_t target_area_localID;
		std::int64_t weight;
	} AreaConnectivityInfo;
};
