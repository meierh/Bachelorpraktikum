#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cassert>  // debug
#include <chrono>

class GraphProperty {
public:

    /*|||------------------areaConnectivityStrength--------------------------
     *
     * Foreach combination of areas A and B, the function sums the weight
     * of all edges connecting a node in area A with a node in area B.
     * 
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     * resultToRank:    MPI Rank to receive the results
     *
     * Returns: std::unordered_map with the a pair of area names as key 
     *          and the summed weight as value. 
     *          (std::pair(area_name_A,area_name_B)->summed_weight) 
     */
    struct stdPair_hash
    {
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const 
        {
            return std::hash<T1>{}(p.first) ^  std::hash<T2>{}(p.second);
        }
    };    
    using AreaConnecMap = std::unordered_map<std::pair<std::string,std::string>,int,stdPair_hash>;
    static std::unique_ptr<AreaConnecMap> areaConnectivityStrength(const DistributedGraph& graph,unsigned int resultToRank=0);
    static std::unique_ptr<AreaConnecMap> areaConnectivityStrengthSingleProc_Helge(const DistributedGraph& graph,unsigned int resultToRank=0);    
    static std::unique_ptr<AreaConnecMap> areaConnectivityStrengthSingleProc(const DistributedGraph& graph,unsigned int resultToRank=0);
    /*------------------areaConnectivityStrength--------------------------|||*/
    
    
    static bool compare_area_connecs(std::unique_ptr<AreaConnecMap> const &map1, std::unique_ptr<AreaConnecMap> const &map2, unsigned int resultToRank=0);
    static bool compare_area_connecs_alt(std::unique_ptr<AreaConnecMap> const &map1, std::unique_ptr<AreaConnecMap> const &map2, unsigned int resultToRank=0);


    /*|||-----------------------Histogram--------------------------------
     *
     * Functions to compute the length of all edges and to count them in 
     * a length histogram
     *
     * Returns: Histogram {std::vector of pairs with the bin borders in 
     *                     the first and the count of edges in this bin}
     */
    using Histogram = std::vector<std::pair<std::pair<double,double>,std::uint64_t>>;
    /*
     * Version of Histogram method that creates a histogram with a given
     * width of bins 
     * 
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     * bin_width:       Width of the bin in the resulting histogram
     * resultToRank:    MPI Rank to receive the results
     */
    static std::unique_ptr<Histogram> edgeLengthHistogramm_constBinWidth
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
    static std::unique_ptr<Histogram> edgeLengthHistogramm_constBinCount
    (
        const DistributedGraph& graph,
        const std::uint64_t bin_count,
        const unsigned int resultToRank=0
    );
    static std::unique_ptr<Histogram> edgeLengthHistogramm_constBinWidthSingleProc
    (
        const DistributedGraph& graph,
        double bin_width,
        unsigned int resultToRank=0
    );
    static std::unique_ptr<Histogram> edgeLengthHistogramm_constBinCountSingleProc
    (
        const DistributedGraph& graph,
        std::uint64_t bin_count,
        unsigned int resultToRank=0
    );
    /*-------------------------Histogram----------------------------------|||*/
    
    static std::vector<long double> networkTripleMotifs
    (
        const DistributedGraph& graph,
        unsigned int resultToRank = 0
    );
    
    /*|||-----------------------Modularity--------------------------------
     *
     * Functions to compute the modularity of the graph
     *
     * Returns: The Modularity according to paper
     *  "A tutorial in connectome analysis: Topological and spatial features of brain networks"
     *  by Marcus Kaiser, 2011 in NeuroImage, (892-907)
     * 
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     */
    static double computeModularity
    (
        const DistributedGraph& graph
    );
    static double computeModularitySingleProc
    (
        const DistributedGraph& graph
    );
    /*-------------------------Modularity----------------------------------|||*/
    
private:
    using AreaLocalID = std::pair<std::uint64_t,std::uint64_t>;
    struct stdDoublePair_hash
    {
        stdPair_hash hash;
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const 
        {
            return hash(p.first) ^  hash(p.second);
        }
    };
    using AreaIDConnecMap = std::unordered_map<std::pair<AreaLocalID,AreaLocalID>,int,stdDoublePair_hash>;
    typedef struct
    {
        std::int64_t source_rank;
        std::int64_t source_area_localID;
        std::int64_t target_rank;
        std::int64_t target_area_localID;
        std::int64_t weight;        
    } areaConnectivityInfo;    
    
    static inline unsigned int cantorPair(unsigned int k1, unsigned int k2) {return (((k1+k2)*(k1+k2+1))/2)+k2;}    
    
    static std::unique_ptr<Histogram> edgeLengthHistogramm
    (
        const DistributedGraph& graph,
        const std::function<std::unique_ptr<Histogram>(const double,const double)> histogram_creator,
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
    static std::unique_ptr<Histogram> edgeLengthHistogramSingleProc
    (
        const DistributedGraph& graph,
        std::function<std::unique_ptr<Histogram>(double,double)> histogram_creator,
        unsigned int resultToRank=0
    );
    
    template<typename Q_parameter,typename A_parameter>
    class NodeToNodeQuestionStructure{
        public:
        //Funktions for use on questioner rank side
            void addQuestionsFromOneNodeToSend
            (
                std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,Q_parameter>>> list_of_adressees_and_parameter,
                std::uint64_t questioner
            );
            void finalizeAddingQuestionsToSend();

        //Funktions for use on answerer rank side
            void setQuestionsReceived
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
            std::vector<int>& get_adressee_ranks_to_nbrOfQuestions();
            std::vector<std::uint64_t>& get_nodes_to_ask_question_for_rank(std::uint64_t rank);
            std::vector<Q_parameter>& get_question_parameters_for_rank(std::uint64_t rank);
            std::vector<A_parameter>& get_answers_for_rank(std::uint64_t ranks);
            
        private:
            // stores index in nodes_to_ask_question for each rank
            std::unordered_map<std::uint64_t,std::uint64_t> rank_to_outerIndex;
            // list of list of nodes to ask questions 
            std::vector<std::vector<std::uint64_t>> nodes_to_ask_question;
            // list of ranks coresponding to nodes_to_ask_question
            std::vector<std::uint64_t> list_index_to_adressee_rank;
            // list of list of nodes that ask the questions coresponding to nodes_to_ask_question
            std::vector<std::vector<std::uint64_t>> nodes_that_ask_the_question;
            // stores index in nodes_to_ask_question for each rank
            std::unordered_multimap<std::uint64_t,std::pair<std::uint64_t,std::uint64_t>> questioner_node_to_outerIndex_and_innerIndex;
            // list of list of parameters coresponding to nodes_to_ask_question
            std::vector<std::vector<Q_parameter>> question_parameters;
            // list of all ranks to number of questions
            std::vector<int> addressee_ranks_to_nbrOfQuestions;
            // list of list of answers coresponding to nodes_to_ask_question
            std::vector<std::vector<A_parameter>> answers_to_questions;
                        
            std::vector<std::uint64_t> dummy_nodes_to_ask_question;
            std::vector<Q_parameter> dummy_question_parameters;
            std::vector<A_parameter> dummy_answers_to_questions;
            
            enum StatusType {Empty,PrepareQuestionsToSend,PrepareQuestionsRecv,ClosedQuestionsPreparation};
            StatusType structureStatus = Empty;
            
            friend GraphProperty;
    };
    
    template<typename Q_parameter,typename A_parameter>
    static std::unique_ptr<NodeToNodeQuestionStructure<Q_parameter,A_parameter>> node_to_node_question
    (
        const DistributedGraph&,
        MPI_Datatype,
        std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,Q_parameter>>>(const DistributedGraph& dg,std::uint64_t node_local_ind)>,
        MPI_Datatype,
        std::function<A_parameter(const DistributedGraph& dg,std::uint64_t node_local_ind,Q_parameter para)>
    );
    
    template<typename DATA,typename DATA_Element>
    static std::unique_ptr<std::vector<std::vector<DATA>>> gather_Data_to_one_Rank
    (
        const DistributedGraph&,
        std::function<std::unique_ptr<std::vector<std::pair<DATA,int>>>(const DistributedGraph& dg)>,
        std::function<std::vector<DATA_Element>(DATA dat)>,
        std::function<DATA(std::vector<DATA_Element>&)>,
        MPI_Datatype,
        int    
    );
    
    template<typename DATA,typename DATA_Element>
    static std::unique_ptr<std::vector<std::vector<DATA>>> gather_Data_to_all_Ranks
    (
        const DistributedGraph&,
        std::function<std::unique_ptr<std::vector<std::pair<DATA,int>>>(const DistributedGraph& dg)>,
        std::function<std::vector<DATA_Element>(DATA dat)>,
        std::function<DATA(std::vector<DATA_Element>&)>,
        MPI_Datatype
    );
    
    template<typename DATA,typename DATA_Element>
    static std::unique_ptr<std::vector<std::vector<DATA>>> gather_Data
    (
        const DistributedGraph&,
        std::function<std::unique_ptr<std::vector<std::pair<DATA,int>>>(const DistributedGraph& )>,
        std::function<std::vector<DATA_Element>(DATA)>,
        std::function<DATA(std::vector<DATA_Element>&)>,
        std::function<void(DATA_Element*,int,DATA_Element*,int*,int*,MPI_Datatype,int)>,
        std::function<void(int*,int,int*,int*,int*,int)>,
        MPI_Datatype,
        int
    );
    
    typedef struct
    {
        std::uint64_t node_1_rank;
        std::uint64_t node_1_local;
        std::uint64_t node_2_rank;
        std::uint64_t node_2_local;
        std::uint64_t node_3_rank;
        std::uint64_t node_3_local;
        std::uint64_t motifTypeBitArray=0;
        
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
        void unsetAllButMotifTypes(std::vector<int> motifTypes)
        {
            std::unordered_set<int> maintainedMotifTypes;
            maintainedMotifTypes.insert(motifTypes.begin(),motifTypes.end());
            std::vector<int> motifTypesToUnset;
            for(int motifType=1;motifType<14;motifType++)
            {
                if(maintainedMotifTypes.find(motifType)==maintainedMotifTypes.end())
                {
                    motifTypesToUnset.push_back(motifType);
                }
            }
            unsetMotifTypes(motifTypesToUnset);
        }
        bool isMotifTypeSet(int motifType)
        {
            assert(motifType>=1 && motifType<14);
            return motifTypeBitArray & (1<<motifType);
        }
        void printOut()
        {
            std::cout<<"-------------";
            for(int i=1;i<14;i++)
                if(isMotifTypeSet(i))
                    std::cout<<"1"<<" ";
                else
                    std::cout<<"0"<<" ";
            std::cout<<"--------------------"<<std::endl;
        }
        bool checkValidity()
        {
            bool res = motifTypeBitArray && !(motifTypeBitArray & (motifTypeBitArray-1));
            if(!res)
            {
                std::cout<<"---------------------";
                for(int i=1;i<13;i++)
                    if(isMotifTypeSet(i))
                        std::cout<<"1"<<" ";
                    else
                        std::cout<<"0"<<" ";
                std::cout<<"--------------------"<<res<<std::endl;
            }
            return res;
        }
    } threeMotifStructure;
    
    typedef struct
    {
        std::uint64_t node_in_degree;
        std::uint64_t node_out_degree;
        std::uint64_t area_global_ID;
    } nodeModularityInfo;
    
};
