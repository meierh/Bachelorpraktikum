#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <cassert>  // debug

class GraphProperty {
public:
    
    /* Foreach combination of areas A and B the function sums the strength
     * of all edges connecting a node in area A with a node in area B.
     * 
     * Parameter: A DistributedGraph (Function is MPI compliant)
     * Return: OPEN  
     */
    struct stdPair_hash
    {
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const 
        {
            return std::hash<T1>{}(p.first) ^  std::hash<T2>{}(p.second);
        }
    };
    struct stdDoublePair_hash
    {
        stdPair_hash hash;
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const 
        {
            return hash(p.first) ^  hash(p.second);
        }
    };    
    using AreaConnecMap = std::unordered_map<std::pair<std::string,std::string>,int,stdPair_hash>;
    using AreaLocalID = std::pair<std::uint64_t,std::uint64_t>;
    using AreaIDConnecMap = std::unordered_map<std::pair<AreaLocalID,AreaLocalID>,int,stdDoublePair_hash>;
    static std::unique_ptr<AreaConnecMap> areaConnectivityStrength(const DistributedGraph& graph,unsigned int resultToRank=0);
    static std::unique_ptr<AreaConnecMap> areaConnectivityStrengthSingleProc(const DistributedGraph& graph,unsigned int resultToRank=0);
    
    /* Histogram for count inside interval greater equal the lower and smaller than the upper bound
     */
    using Histogram = std::vector<std::pair<std::pair<double,double>,std::uint64_t>>;
    static std::unique_ptr<Histogram> edgeLengthHistogramm_constBinWidth
    (
        const DistributedGraph& graph,
        double bin_width,
        unsigned int resultToRank=0
    );
    static std::unique_ptr<Histogram> edgeLengthHistogramm_constBinCount
    (
        const DistributedGraph& graph,
        std::uint64_t bin_count,
        unsigned int resultToRank=0
    );
    
    static std::vector<double> networkTripleMotifs
    (
        const DistributedGraph& graph,
        unsigned int resultToRank = 0
    );
    
    double computeModularity
    (
        const DistributedGraph& graph
    );
    
private:
    static inline unsigned int cantorPair(unsigned int k1, unsigned int k2) {return (((k1+k2)*(k1+k2+1))/2)+k2;}    
    
    /* Shortcuts for collectAlongEdges_InToOut method
     */
    template<typename DATA>
    using collectedData_ptr = std::vector<std::vector<DATA>>;
    using collectedDataStructure_ptr = std::vector<std::unordered_map<std::uint64_t,int>>;
    using collectedDataIndexes_ptr = std::unordered_map<int,std::pair<int,int>>;
    
    /* General method for transfering one date of the type DATA from any in edge to any out edge
     */
    template<typename DATA>
    static std::tuple<
        std::unique_ptr<collectedData_ptr<DATA>>,
        std::unique_ptr<collectedDataStructure_ptr>,
        std::unique_ptr<collectedDataIndexes_ptr>>
    collectAlongEdges_InToOut
    (
        const DistributedGraph& graph,
        MPI_Datatype datatype,
        std::function<DATA(int,int)> date_get_function
    );
    
    static std::unique_ptr<Histogram> edgeLengthHistogramm
    (
        const DistributedGraph& graph,
        std::function<std::unique_ptr<Histogram>(double,double)> histogram_creator,
        unsigned int resultToRank
    );
    
    template<typename Q_parameter,typename A_parameter>
    class NodeToNodeQuestionStructure{
        public:
            void addAddresseesAndParameterFromOneNode
            (
                std::vector<std::tuple<std::uint64_t,std::uint64_t,Q_parameter>> list_of_adressees_and_parameter,
                std::uint64_t questioner
            );
            void setQuestionsAndParameterRecv
            (
                std::vector<std::uint64_t>& total_nodes_to_ask_question,
                std::vector<Q_parameter>& total_question_parameters,
                std::vector<int>& rank_size,
                std::vector<int>& rank_displ
            );
            void computeAnswersToQuestions
            (
                const DistributedGraph& dg,
                std::function<A_parameter(const DistributedGraph& dg,std::uint64_t node_local_ind,Q_parameter para)> generateAnswers
            );
            void setAnswers
            (
                std::vector<A_parameter>& total_answers,
                std::vector<int>& rank_size,
                std::vector<int>& rank_displ
            );
            std::unique_ptr<std::vector<A_parameter>> getAnswersOfQuestionerNode
            (
                std::uint64_t node_local_ind
            );
            std::vector<int>& get_ranks_to_nbrOfQuestions(const int number_ranks);
            std::vector<std::uint64_t>& get_nodes_to_ask_question_for_rank(std::uint64_t rank);
            std::vector<Q_parameter>& get_question_parameters_for_rank(std::uint64_t rank);
            std::vector<A_parameter>& get_answers_for_rank(std::uint64_t ranks);
        private:
            // stores index in nodes_to_ask_question for each rank
            std::unordered_map<std::uint64_t,std::uint64_t> rank_to_outerIndex;
            // list of list of nodes to ask questions 
            std::vector<std::vector<std::uint64_t>> nodes_to_ask_question;
            // list of ranks coresponding to nodes_to_ask_question
            std::vector<std::uint64_t> rank_of_list_index;
            // list of list of nodes that ask the questions coresponding to nodes_to_ask_question
            std::vector<std::vector<std::uint64_t>> nodes_that_ask_the_question;
            // stores index in nodes_to_ask_question for each rank
            std::unordered_multimap<std::uint64_t,std::pair<std::uint64_t,std::uint64_t>> questioner_node_to_outerIndex_and_innerIndex;
            // list of list of parameters coresponding to nodes_to_ask_question
            std::vector<std::vector<Q_parameter>> question_parameters;
            // list of all ranks to number of questions
            std::vector<int> ranks_to_nbrOfQuestions;
            // list of list of answers coresponding to nodes_to_ask_question
            std::vector<std::vector<A_parameter>> answers_to_questions;
            
            std::vector<std::uint64_t> dummy_nodes_to_ask_question;
            std::vector<Q_parameter> dummy_question_parameters;
            std::vector<A_parameter> dummy_answers_to_questions;

    };
    
    template<typename Q_parameter,typename A_parameter>
    static std::unique_ptr<NodeToNodeQuestionStructure<Q_parameter,A_parameter>> node_to_node_question
    (
        const DistributedGraph& graph,
        MPI_Datatype MPI_Q_parameter,
        std::function<std::vector<std::tuple<std::uint64_t,std::uint64_t,Q_parameter>>(const DistributedGraph& dg,std::uint64_t node_local_ind)> generateAddressees,
        MPI_Datatype MPI_A_parameter,
        std::function<A_parameter(const DistributedGraph& dg,std::uint64_t node_local_ind,Q_parameter para)> generateAnswers
    );
    
    typedef struct
    {
        std::uint64_t node_1_rank;
        std::uint64_t node_1_local;
        std::uint64_t node_2_rank;
        std::uint64_t node_2_local;
        std::uint64_t node_3_rank;
        std::uint64_t node_3_local;
        std::uint16_t motifTypeBitArray;
        
        void setMotifTypes(std::vector<int> motifTypes)
        {
            for(int motifType : motifTypes)
            {
                assert(motifType>=1 && motifType<14);
                motifTypeBitArray |= (1<<motifType);
            }
        }
        void unsetMotifTypes(std::vector<int> motifTypes)
        {
            for(int motifType : motifTypes)
            {
                assert(motifType>=1 && motifType<14);
                motifTypeBitArray &= ~(1<<motifType);
            }
        }
        bool isMotifTypeSet(int motifType)
        {
            assert(motifType>=1 && motifType<14);
            return motifTypeBitArray & (1<<motifType);
        }
        bool checkValidity()
        {
            return motifTypeBitArray && !(motifTypeBitArray & (motifTypeBitArray-1));
        }
    } threeMotifStructure;
    
    typedef struct
    {
        std::uint64_t node_in_degree;
        std::uint64_t node_out_degree;
        std::uint64_t area_global_ID;
    } nodeModularityInfo;
    
};
