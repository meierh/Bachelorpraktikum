#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <cassert>
#include "CommunicationPatterns.h"
#include <numeric>


/*
 * --------------- General ---------------
 * 
 * This implementation requires that the provided graph
 * - has no nodes with self-referencing edges
 * - has no edge duplicates
 * Otherwise, it can lead to incorrect results or error throwing.
 *
 *	distributed_8 (old format):
 *	- self-referencing edges exist (e.g. distributed_8: rank_0_in_edges.txt, Line 223)
 *
 *	large_graphs (neues format):
 *	- edge-duplicates exist (e.g. largeGraph: 32/network/rank_00_out_network.txt, Line 7-8)
 *
 *      => This properties can be analyzed with the AlgorithmTests::check_graph_property() method
*/

class Assortativity {
public:
	/*|||--------------compute_assortativity_coefficient------------------
	 * Computes the assortativity coefficients inspired by the Paper
     * "Assortative mixing in networks" by Newman Equation 3
     * 
     * WARNING: The formula for the joint degree distribution could not
     *          be applied as it was in the paper due to the fact that the 
     *          paper treated undirected graphs.
     *          Adaptions were made here.
     * 
     *          Cases with a zero expectation or a zero std deviation 
     *          were treated specially to avoid zero division.
     *          As a consequence the results may not be as expected
     * 
	 * Parameters
	 * graph:           A DistributedGraph (Function is MPI compliant)
	 *
	 * Returns: A tuple with the coefficients ordered a follows
     *          std::tuple<r_in_in,r_in_out,r_out_in,r_out_out>
     * 
     * MPI Constraints:  Function must be called by every rank simultaneously
     *                   Function returns correct result to all ranks
	 */
    static std::tuple<double,double,double,double> compute_assortativity_coefficient
    (
        const DistributedGraph& graph
    );
	/*|||--------------compute_assortativity_coefficient------------------*/

    static std::tuple<double,double,double,double> compute_assortativity_coefficient_sequential
    (
        const DistributedGraph& graph
    );

    static void compare_Parts
    (
        const DistributedGraph& graph
    );
    
private:
    /*|||--------------compute_max_nodeDegree_OutIn------------------
	 * Computes the maximum degree of any node in the graph
     * 
	 * Parameters
	 * graph:           A DistributedGraph (Function is MPI compliant)
	 *
	 * Returns: A pair with the maximum node degree ordered a follows
     *          std::pair<max_out_degree,max_in_degree>
     * 
     * MPI Constraints:  Function must be called by every rank simultaneously
     *                   Function returns correct result to all ranks
	 */
    static std::pair<std::uint64_t,std::uint64_t> compute_max_nodeDegree_OutIn
    (
        const DistributedGraph& graph
    );
    /*|||--------------compute_max_nodeDegree_OutIn------------------*/

    /*|||--------------compute_node_degree_distribution------------------
	 * Computes a distribution of the the node degree in the graph
     * 
	 * Parameters
	 * graph:           A DistributedGraph (Function is MPI compliant)
	 *
	 * Returns: A pair with the node degree distribution ordered a follows
     *          std::pair<out_degree_distribution,in_degree_distribution>
     *          The distributions are vectors of probabilities so that
     *          probability(node_degree=x) = vector[x]
     * 
     * MPI Constraints:  Function must be called by every rank simultaneously
     *                   Function returns correct result to all ranks
	 */
    static std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> compute_node_degree_distribution
    (
        const DistributedGraph& graph
    );
    /*|||--------------compute_node_degree_distribution------------------*/

	/*|||----------compute_normalized_node_degree_distribution------------
	 * Computes a normalized distribution of the the node degree in the graph
     * according to 
     * "Assortative mixing in networks" by Newman Equation 1
     * 
	 * Parameters
	 * graph:           A DistributedGraph (Function is MPI compliant)
	 *
	 * Returns: A pair with the normalized node degree distribution ordered a follows
     *          std::pair<out_degree_distribution,in_degree_distribution>
     *          The distributions are vectors of probabilities so that
     *          probability(node_degree=x) = vector[x]
     * 
     * MPI Constraints:  Function must be called by every rank simultaneously
     *                   Function returns correct result to all ranks
	 */
    static std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> compute_normalized_node_degree_distribution
    (
        const DistributedGraph& graph
    );
	/*|||----------compute_normalized_node_degree_distribution------------*/
    

	/*|||-----compute_standard_deviation_of_node_degree_distribution--------
	 * Computes the standard deviation of a pair of distributions
     * 
	 * Parameters
	 * graph:           A DistributedGraph (Function is MPI compliant)
     * node_degree_distribution: A pair of two distributions
     *                          If the entries in any vector does not fulfill
     *                          The constraints of a distribution the results 
     *                          are not guaranteed
     * 
	 * Returns: A pair with the standard deviation for the distributions in
     *          the same order as given in the parameter
	 */
    static std::pair<double,double> compute_standard_deviation_of_node_degree_distribution
    (
        const std::unique_ptr<std::pair<std::vector<double>,
        std::vector<double>>>& node_degree_distribution
    );
	/*|||-----compute_standard_deviation_of_node_degree_distribution--------*/

    
    typedef struct
    {
        std::uint64_t source_OutDegree;
        std::uint64_t source_InDegree;
        std::uint64_t target_OutDegree;
        std::uint64_t target_InDegree;
    } EdgeDegrees;    
    
    /* General 2D Distribution class
     * Methods act according to their name
     */
    template<typename T>
    class Distribution2D{
        public:
            Distribution2D(int first_dimension, int second_dimension);
            Distribution2D();
            
            void set_probability(int first_index, int second_index, T probability);
            T get_probability(int first_index, int second_index);
            int get_first_dimension(){return first_dimension;};
            int get_second_dimension(){return second_dimension;};
            void reset(int first_dimension, int second_dimension);
            std::vector<T>& data() {return probabilities;};
            void operate_on_index(int first_index,int second_index,std::function<T(T)> operation);
            void operate_on_all(std::function<T(int,int,T)> operation);
            
            /* Dont use the following methods to extract or insert data if the same  
             * operation is possible using other methods
             */
            T* data_ptr() {return probabilities.data();};
            T compute_Expectation();
            void print(){for(T d:probabilities) std::cout<<" "<<d; std::cout<<std::endl;};
        
        private:
            int first_dimension;
            int second_dimension;
            std::vector<T> probabilities;
    };    
    
    /*|||-----compute_joint_edge_degree_distribution--------
	 * Computes the normalized joint edge degree distribution
     * 
     * The computation was adapted to be normalized to allow for
     * the computation in
     * "Assortative mixing in networks" by Newman Equation 3
     * as a result the distributions are normalized in the same was as
     * it was done in equation 1 of the stated paper only with an 
     * adaption for 2D distributions
     * 
	 * Parameters
	 * graph:           A DistributedGraph (Function is MPI compliant)
     * 
	 * Returns: A tuple of the 2D distributions 
     *          std::tuple<e_in_in,e_in_out,e_out_in,e_out_out>
	 */
    static std::unique_ptr<std::tuple<Distribution2D<double>,Distribution2D<double>,Distribution2D<double>,Distribution2D<double>>> compute_joint_edge_degree_distribution
    (
        const DistributedGraph& graph
    );
    /*|||-----compute_joint_edge_degree_distribution--------*/

    
    static std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> compute_node_degree_distribution_sequential
    (    
        const DistributedGraph& graph
    );
    static std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>> compute_normalized_node_degree_distribution_sequential
    (    
        const DistributedGraph& graph
    );
    static std::pair<double,double> compute_standard_deviation_of_node_degree_distribution_sequential
    (
        const std::unique_ptr<std::pair<std::vector<double>,std::vector<double>>>& node_degree_distribution
    );
    static std::unique_ptr<std::tuple<Distribution2D<double>,Distribution2D<double>,Distribution2D<double>,Distribution2D<double>>> compute_joint_edge_degree_distribution_sequential
    (
        const DistributedGraph& graph
    );
};
