#pragma once

#include "MPIWrapper.h"

#include <cstdint>
#include <iostream>

class Status {
public:
	static void report_status(std::uint64_t current_local_value, std::uint64_t total_value, std::string algorithm_name) {
		const auto my_rank = MPIWrapper::get_my_rank();

		const auto total_local_values = MPIWrapper::reduce_sum(current_local_value);

		if (my_rank != 0) {
			return;
		}

		std::cout << "Status report on " << algorithm_name << '\n';
		std::cout << "Processed a total of " << total_local_values << " of " << total_value << " iterations.\n";
		fflush(stdout);
	}
};
