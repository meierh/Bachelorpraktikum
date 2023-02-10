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

//Forward declaration
template<typename Q_parameter,typename A_parameter>
class NodeToNodeQuestionStructure;

class CommunicationPatterns {
public:  
    /*|||------------------node_to_node_question--------------------------
     *
     * 
     * 
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     * ...
     *
     * Returns: ...
     */    
    template<typename Q_parameter,typename A_parameter>
    static std::unique_ptr<NodeToNodeQuestionStructure<Q_parameter,A_parameter>> node_to_node_question
    (
        const DistributedGraph&,
        MPI_Datatype,
        std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,Q_parameter>>>(const DistributedGraph& dg,std::uint64_t node_local_ind)>,
        MPI_Datatype,
        std::function<A_parameter(const DistributedGraph& dg,std::uint64_t node_local_ind,Q_parameter para)>
    );
    /*---------------------node_to_node_question--------------------------|||*/

    /*|||------------------gather_Data_to_one_Rank--------------------------
     *
     * 
     * 
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     * ...
     *
     * Returns: ...
     */
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
    /*---------------------gather_Data_to_one_Rank--------------------------|||*/
    
    /*|||------------------gather_Data_to_all_Ranks--------------------------
     *
     * 
     * 
     * Parameters 
     * graph:           A DistributedGraph (Function is MPI compliant)
     * ...
     *
     * Returns: ...
     */
    template<typename DATA,typename DATA_Element>
    static std::unique_ptr<std::vector<std::vector<DATA>>> gather_Data_to_all_Ranks
    (
        const DistributedGraph&,
        std::function<std::unique_ptr<std::vector<std::pair<DATA,int>>>(const DistributedGraph& dg)>,
        std::function<std::vector<DATA_Element>(DATA dat)>,
        std::function<DATA(std::vector<DATA_Element>&)>,
        MPI_Datatype
    );
    /*---------------------gather_Data_to_all_Ranks--------------------------|||*/
    
private:
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
};

template<typename Q_parameter,typename A_parameter>
class NodeToNodeQuestionStructure{
public:
//Functions for use on questioner rank side
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
    
    friend CommunicationPatterns;
};
