#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include "CommunicationPatterns.h"
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <cassert>

class Histogram {
public:
    /*|||-----------------------Histogram--------------------------------
     *
     * Functions to compute the length of all edges and to count them in 
     * a length histogram
     *
     * Returns: Histogram {std::vector of pairs with the bin borders in 
     *                     the first and the count of edges in this bin}
     */
    using HistogramData = std::vector<std::pair<std::pair<double,double>,std::uint64_t>>;
    /*
     * Version of Histogram method that creates a histogram with a given
     * width of bins 
     * 
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     * bin_width:       Width of the bin in the resulting histogram
     * resultToRank:    MPI Rank to receive the results
     */
    static std::unique_ptr<HistogramData> edgeLengthHistogramm_constBinWidth
    (
        const DistributedGraph& graph,
        const double bin_width,
        const unsigned int resultToRank=0
    );
    /*
     * Version of Histogram method that creates a histogram with a given
     * number of bins 
     * 
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     * bin_count:       Number of bins in the resulting histogram
     * resultToRank:    MPI Rank to receive the results
     */
    static std::unique_ptr<HistogramData> edgeLengthHistogramm_constBinCount
    (
        const DistributedGraph& graph,
        const std::uint64_t bin_count,
        const unsigned int resultToRank=0
    );
    static std::unique_ptr<HistogramData> edgeLengthHistogramm_constBinWidthSingleProc
    (
        const DistributedGraph& graph,
        double bin_width,
        unsigned int resultToRank=0
    );
    static std::unique_ptr<HistogramData> edgeLengthHistogramm_constBinCountSingleProc
    (
        const DistributedGraph& graph,
        std::uint64_t bin_count,
        unsigned int resultToRank=0
    );
    /*-------------------------Histogram----------------------------------|||*/

private:
    static std::unique_ptr<HistogramData> edgeLengthHistogramm
    (
        const DistributedGraph& graph,
        const std::function<std::unique_ptr<HistogramData>(const double,const double)> histogram_creator,
        const unsigned int resultToRank
    );

    /*
    * This single process variant of the edgeLengthHistogram method lets only one main process
    * collect and store all edge lengths in the resulting histogram.
    * 
    * @param graph underlaying graph
    * @param histogram_creator function to crate a histogram in a specified shape
    * @param resultToRank main process for computation
    * @returns unique_ptr of the histogram created using the histogram_creater function
    */
    static std::unique_ptr<HistogramData> edgeLengthHistogramSingleProc
    (
        const DistributedGraph& graph,
        std::function<std::unique_ptr<HistogramData>(double,double)> histogram_creator,
        unsigned int resultToRank=0
    );
};
