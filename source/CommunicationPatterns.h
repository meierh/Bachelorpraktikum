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

class CommunicationPatterns {    
public:
    //Forward declaration
    template<typename Q_parameter,typename A_parameter>
    class NodeToNodeQuestionStructure;
    
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
        const DistributedGraph& graph,
        MPI_Datatype MPI_Q_parameter,
        std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,Q_parameter>>>(const DistributedGraph& dg,std::uint64_t node_local_ind)> generateAddressees,
        MPI_Datatype MPI_A_parameter,
        std::function<A_parameter(const DistributedGraph& dg,std::uint64_t node_local_ind,Q_parameter para)> generateAnswers
    )
    {
        std::vector<double> times;
        std::vector<std::string> code_names = {"GenQuestion","DistrNumbers","DistrQuestions","SetQuestions","CompAnswers",
                                                "SendAnswers","SetAnswers"};
        auto start = std::chrono::steady_clock::now(); 

        const int my_rank = MPIWrapper::get_my_rank();
        const int number_ranks = MPIWrapper::get_number_ranks();
        const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
        
    // Collect Questions and create Questioners structure
        auto questioner_structure = std::make_unique<NodeToNodeQuestionStructure<Q_parameter,A_parameter>>();
        for(std::uint64_t node_local_ind=0;node_local_ind<number_local_nodes;node_local_ind++)
        {
            questioner_structure->addQuestionsFromOneNodeToSend(generateAddressees(graph,node_local_ind),node_local_ind);
        }
        questioner_structure->finalizeAddingQuestionsToSend();
        
        times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
        start = std::chrono::steady_clock::now(); 
                
    // Distribute number of questions to each rank
        std::vector<int>& send_ranks_to_nbrOfQuestions = questioner_structure->get_adressee_ranks_to_nbrOfQuestions();
        std::vector<int> global_ranks_to_nbrOfQuestions(number_ranks*number_ranks);    
        std::vector<int> destCounts_ranks_to_nbrOfQuestions(number_ranks,number_ranks);
        std::vector<int> displ_ranks_to_nbrOfQuestions(number_ranks);
        for(int index = 0;index<displ_ranks_to_nbrOfQuestions.size();index++)
        {
            displ_ranks_to_nbrOfQuestions[index]=number_ranks*index;
        }
        MPIWrapper::all_gatherv<int>(send_ranks_to_nbrOfQuestions.data(), number_ranks,
                                    global_ranks_to_nbrOfQuestions.data(), destCounts_ranks_to_nbrOfQuestions.data(), displ_ranks_to_nbrOfQuestions.data(), MPI_INT);

        times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
        start = std::chrono::steady_clock::now(); 
        
    // Distribute questions to each rank
        std::vector<int> recv_ranks_to_nbrOfQuestions(number_ranks);
        for(int rank=0;rank<recv_ranks_to_nbrOfQuestions.size();rank++)
        {
            recv_ranks_to_nbrOfQuestions[rank] = global_ranks_to_nbrOfQuestions[rank*number_ranks+my_rank];
        }
        std::vector<int> displ_recv_ranks_to_nbrOfQuestions(number_ranks,0);
        for(int index = 1;index<displ_recv_ranks_to_nbrOfQuestions.size();index++)
        {
            displ_recv_ranks_to_nbrOfQuestions[index]=
                displ_recv_ranks_to_nbrOfQuestions[index-1]+recv_ranks_to_nbrOfQuestions[index-1];
        }
        int my_rank_total_receive_size = std::accumulate(recv_ranks_to_nbrOfQuestions.begin(),
                                                        recv_ranks_to_nbrOfQuestions.end(),0);
        std::vector<std::uint64_t> my_rank_total_nodes_to_ask_question(my_rank_total_receive_size);
        std::vector<Q_parameter> my_rank_total_question_parameters(my_rank_total_receive_size);
        for(int rank=0;rank<number_ranks;rank++)
        {
            std::vector<std::uint64_t>& nodes_to_ask_question_for_rank =
                questioner_structure->get_nodes_to_ask_question_for_rank(rank);
            std::vector<Q_parameter>& question_parameters_for_rank =
                questioner_structure->get_question_parameters_for_rank(rank);
            assert(nodes_to_ask_question_for_rank.size()==question_parameters_for_rank.size());
            int count = nodes_to_ask_question_for_rank.size();
            
            MPIWrapper::gatherv<std::uint64_t>(nodes_to_ask_question_for_rank.data(), count,
                                            my_rank_total_nodes_to_ask_question.data(),
                                            recv_ranks_to_nbrOfQuestions.data(), displ_recv_ranks_to_nbrOfQuestions.data(),MPI_UINT64_T,rank);
            
            MPIWrapper::gatherv<Q_parameter>(question_parameters_for_rank.data(), count,
                                            my_rank_total_question_parameters.data(),
                                            recv_ranks_to_nbrOfQuestions.data(), displ_recv_ranks_to_nbrOfQuestions.data(),MPI_Q_parameter,rank);
        }
        
        times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
        start = std::chrono::steady_clock::now(); 

    // Set questions to be answered to adressees questioner structure
        NodeToNodeQuestionStructure<Q_parameter,A_parameter> adressee_structure;
        adressee_structure.setQuestionsReceived(my_rank_total_nodes_to_ask_question,my_rank_total_question_parameters,
                                                recv_ranks_to_nbrOfQuestions,displ_recv_ranks_to_nbrOfQuestions);
        
        times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
        start = std::chrono::steady_clock::now(); 
        
    // Compute the answers to questions
        adressee_structure.computeAnswersToQuestions(graph,generateAnswers);
        
        times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
        start = std::chrono::steady_clock::now(); 
        
    // Send answers back to questioners
        std::vector<int>& send_ranks_to_nbrOfAnswers =  questioner_structure->get_adressee_ranks_to_nbrOfQuestions();
        std::vector<int> displ_send_ranks_to_nbrOfAnswers(number_ranks,0);
        for(int index = 1;index<displ_send_ranks_to_nbrOfAnswers.size();index++)
        {
            displ_send_ranks_to_nbrOfAnswers[index]=
                displ_send_ranks_to_nbrOfAnswers[index-1]+send_ranks_to_nbrOfAnswers[index-1];
        }
        my_rank_total_receive_size = std::accumulate(send_ranks_to_nbrOfAnswers.begin(),
                                                    send_ranks_to_nbrOfAnswers.end(),0);
        std::vector<A_parameter> my_rank_total_answer_parameters(my_rank_total_receive_size);
        for(int rank=0;rank<number_ranks;rank++)
        {
            std::vector<A_parameter>& answers_for_rank = adressee_structure.get_answers_for_rank(rank);
            
            int count = answers_for_rank.size();        
            
            A_parameter* intermed = answers_for_rank.data();
            
            MPIWrapper::gatherv<A_parameter>(intermed, count,
                                            my_rank_total_answer_parameters.data(),
                                            send_ranks_to_nbrOfAnswers.data(), displ_send_ranks_to_nbrOfAnswers.data(),MPI_A_parameter,rank);
        }
        
        times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
        start = std::chrono::steady_clock::now(); 
        
    // Set answers questioner structure
        questioner_structure->setAnswers(my_rank_total_answer_parameters,send_ranks_to_nbrOfAnswers,
                                        displ_send_ranks_to_nbrOfAnswers);

        times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
        start = std::chrono::steady_clock::now(); 
        
        std::cout<<"Rank:"<<my_rank<<"[ ";
        for(int i=0;i<times.size();i++)
        {
            std::cout<<" "<<code_names[i]<<":"<<times[i]<<" ";
        }
        std::cout<<"]"<<std::endl;
        
        return std::move(questioner_structure);
    };
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
        const DistributedGraph& dg,
        std::function<std::unique_ptr<std::vector<std::pair<DATA,int>>>(const DistributedGraph&)> getData,
        std::function<std::vector<DATA_Element>(DATA)> transformDataToElementary,
        std::function<DATA(std::vector<DATA_Element>&)> transformElementaryToData,
        MPI_Datatype DATA_Element_datatype,
        int root
    )
    {
        auto dataGatherMethod=[](DATA_Element* src, int count, DATA_Element* dest, int* destCounts,
                                int* displs, MPI_Datatype datatype,int root)
        {
            MPIWrapper::gatherv<DATA_Element>(src,count,dest,destCounts,displs,datatype,root);
        };
        
        auto sizesGatherMethod=[](int* src, int count, int* dest, int* destCounts,
                                int* displs, int root)
        {
            MPIWrapper::gatherv<int>(src,count,dest,destCounts,displs,MPI_INT,root);
        };

        return std::move(gather_Data<DATA,DATA_Element>(
            dg,getData,transformDataToElementary,transformElementaryToData,
            dataGatherMethod,sizesGatherMethod,DATA_Element_datatype,root
        ));
    };
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
        const DistributedGraph& dg,
        std::function<std::unique_ptr<std::vector<std::pair<DATA,int>>>(const DistributedGraph&)> getData,
        std::function<std::vector<DATA_Element>(DATA)> transformDataToElementary,
        std::function<DATA(std::vector<DATA_Element>&)> transformElementaryToData,
        MPI_Datatype DATA_Element_datatype
    )
    { 
        auto dataGatherMethod=[](DATA_Element* src, int count, DATA_Element* dest, int* destCounts,
                                int* displs, MPI_Datatype datatype,int root)
        {
            MPIWrapper::all_gatherv<DATA_Element>(src,count,dest,destCounts,displs,datatype);
        };
        
        auto sizesGatherMethod=[](int* src, int count, int* dest, int* destCounts,
                                int* displs, int root)
        {
            MPIWrapper::all_gatherv<int>(src,count,dest,destCounts,displs,MPI_INT);
        };

        return std::move(gather_Data<DATA,DATA_Element>(
            dg,getData,transformDataToElementary,transformElementaryToData,dataGatherMethod,
            sizesGatherMethod,DATA_Element_datatype,-1)
        );
    };
    /*---------------------gather_Data_to_all_Ranks--------------------------|||*/
    
private:
    template<typename DATA,typename DATA_Element>
    static std::unique_ptr<std::vector<std::vector<DATA>>> gather_Data
    (
        const DistributedGraph& dg,
        std::function<std::unique_ptr<std::vector<std::pair<DATA,int>>>(const DistributedGraph& dg)> getData,
        std::function<std::vector<DATA_Element>(DATA dat)> transformDataToElementary,
        std::function<DATA(std::vector<DATA_Element>&)> transformElementaryToData,
        std::function<void(DATA_Element* src, int count, DATA_Element* dest, int* destCounts, int* displs, MPI_Datatype datatype,int root)> dataGatherMethod,
        std::function<void(int* src, int count, int* dest, int* destCounts, int* displs,int root)> sizesGatherMethod,
        MPI_Datatype DATA_Element_datatype,
        int root
    )
    {
        const int my_rank = MPIWrapper::get_my_rank();
        const int number_ranks = MPIWrapper::get_number_ranks();
        int data_target_size = (root==-1 || root==my_rank)?number_ranks:0;
        
        //Generate and transform data
        std::unique_ptr<std::vector<std::pair<DATA,int>>> data = getData(dg);
        int local_number_data = data->size();
        std::vector<int> data_inner_size(local_number_data);
        std::transform(data->begin(),data->end(),data_inner_size.begin(),
                    [](std::pair<DATA,int> p){return p.second;});
        std::vector<DATA_Element> local_DATA_Elements;
        std::for_each(data->cbegin(),data->cend(),
                    [&](std::pair<DATA,int> p)
                        { 
                            std::vector<DATA_Element> vec = transformDataToElementary(p.first);
                            local_DATA_Elements.insert(local_DATA_Elements.end(),vec.begin(),vec.end());
                        });
        int local_DATA_Elements_size = local_DATA_Elements.size();
        
        //Gather number of DATA items
        std::vector<int> global_local_number_data(data_target_size);
        std::vector<int> destCountNbr(data_target_size,1);
        std::vector<int> displsNbr(data_target_size,0);
        if(root==-1 || root==my_rank)
        {
            std::partial_sum(destCountNbr.begin(), destCountNbr.end()-1,
                            displsNbr.begin()+1, std::plus<int>());
        }
        sizesGatherMethod(&local_number_data,1,global_local_number_data.data(),destCountNbr.data(),
                        displsNbr.data(),root);
        
        //Gather inner size of data elements
        std::vector<int> global_data_inner_size;
        std::vector<int> destCountInnerNbr(data_target_size);
        std::vector<int> displsInnerNbr(data_target_size,0);
        if(root==-1 || root==my_rank)
        {
            global_data_inner_size.resize(std::accumulate(global_local_number_data.cbegin(),
                                                        global_local_number_data.cend(),0));
            std::transform(global_local_number_data.begin(),global_local_number_data.end(),
                        destCountInnerNbr.begin(),[](int len){return len;});
            std::partial_sum(destCountInnerNbr.begin(), destCountInnerNbr.end()-1,
                            displsInnerNbr.begin()+1, std::plus<int>());
        }
        sizesGatherMethod(data_inner_size.data(),local_number_data,global_data_inner_size.data(),
                        destCountInnerNbr.data(),displsInnerNbr.data(),root);
        
        //Gather DATA_Elements
        std::vector<DATA_Element> global_DATA_Elements;
        std::vector<int> destCountDATAElements(data_target_size);
        std::vector<int> displsInnerDATAElements(data_target_size,0);
        if(root==-1 || root==my_rank)
        {
            global_DATA_Elements.resize(std::accumulate(global_data_inner_size.begin(),
                                                        global_data_inner_size.end(),0));
            int index=0;
            std::transform(global_local_number_data.begin(),global_local_number_data.end(),
                        destCountDATAElements.begin(),[&](int len)
                            { 
                                int count = std::accumulate(global_data_inner_size.begin()+index,global_data_inner_size.begin()+index+len,0);
                                index+=len;
                                return count;
                            });
            std::partial_sum(destCountDATAElements.begin(),destCountDATAElements.end()-1,
                            displsInnerDATAElements.begin()+1,std::plus<int>());
        }
        dataGatherMethod(local_DATA_Elements.data(),local_DATA_Elements_size,
                        global_DATA_Elements.data(),destCountDATAElements.data(),
                        displsInnerDATAElements.data(),DATA_Element_datatype,root);

        //Reorganize DATA
        auto collectedData = std::make_unique<std::vector<std::vector<DATA>>>();
        if(root==-1 || root==my_rank)
        {
            collectedData->resize(number_ranks);
            for(int rank=0;rank<number_ranks;rank++)
            {
                int rank_DataElement_Start = displsInnerDATAElements[rank];
                int rank_number_data = destCountInnerNbr[rank];
                std::vector<int> rank_innerSize(rank_number_data);
                std::transform(global_data_inner_size.begin()+displsInnerNbr[rank],
                            global_data_inner_size.begin()+displsInnerNbr[rank]+destCountInnerNbr[rank],
                            rank_innerSize.begin(),[](int len){return len;});
                int displacement=0;
                for(int j=0;j<rank_innerSize.size();j++)
                {
                    std::vector<DATA_Element> dat(rank_innerSize[j]);
                    std::memcpy(dat.data(),&global_DATA_Elements[rank_DataElement_Start+displacement],
                                rank_innerSize[j]*sizeof(DATA_Element));
                    displacement+=rank_innerSize[j];
                    (*collectedData)[rank].push_back(transformElementaryToData(dat));
                }
            }
        }
        return std::move(collectedData);
    };
    
public:
    template<typename Q_parameter,typename A_parameter>
    class NodeToNodeQuestionStructure{
    public:
    //Functions for use on questioner rank side
        void addQuestionsFromOneNodeToSend
        (
            std::unique_ptr<std::vector<std::tuple<std::uint64_t,std::uint64_t,Q_parameter>>> list_of_adressees_and_parameter,
            std::uint64_t questioner
        )
        {
            assert(structureStatus==Empty || structureStatus==PrepareQuestionsToSend);
            
            for(auto [target_rank,target_local_node,Q_parameter_struct] : *list_of_adressees_and_parameter)
            {
                assert(target_rank<MPIWrapper::get_number_ranks());
                const auto rank_to_index = rank_to_outerIndex.find(target_rank);
                //std::cout<<"target_rank:"<<target_rank<<"  target_local_node:"<<target_local_node<<std::endl;
                if(rank_to_index != rank_to_outerIndex.end())
                // If rank was already encoutered due to other outEdge 
                {
                    std::uint64_t outerIndex = rank_to_index->second;
                    
                    assert(outerIndex>=0);
                    assert(outerIndex<nodes_to_ask_question.size());
                    assert(outerIndex<nodes_that_ask_the_question.size());
                    assert(outerIndex<question_parameters.size());
                    
                    assert(nodes_that_ask_the_question[outerIndex].size()==
                        nodes_to_ask_question[outerIndex].size());
                    assert(nodes_that_ask_the_question[outerIndex].size()==
                        question_parameters[outerIndex].size());
                    
                    std::uint64_t innerIndex = nodes_to_ask_question[outerIndex].size();
                                
                    nodes_to_ask_question[outerIndex].push_back(target_local_node);
                    nodes_that_ask_the_question[outerIndex].push_back(questioner);
                    question_parameters[outerIndex].push_back(Q_parameter_struct);
                                
                    auto doublePair = 
                    std::pair<std::uint64_t,std::pair<std::uint64_t,std::uint64_t>>(questioner,{outerIndex,innerIndex});
                    questioner_node_to_outerIndex_and_innerIndex.insert(doublePair);
                }
                else
                {
                    std::uint64_t outerIndex = nodes_to_ask_question.size();
                    std::uint64_t innerIndex = 0;
                    
                    nodes_to_ask_question.push_back({target_local_node});
                    nodes_that_ask_the_question.push_back({questioner});
                    question_parameters.push_back({Q_parameter_struct});
                    
                    auto doublePair = 
                    std::pair<std::uint64_t,std::pair<std::uint64_t,std::uint64_t>>(questioner,{outerIndex,innerIndex});
                    questioner_node_to_outerIndex_and_innerIndex.insert(doublePair);
                            
                    rank_to_outerIndex.insert({target_rank,outerIndex});
                }
            }
            
            for(auto keyValue = questioner_node_to_outerIndex_and_innerIndex.begin();
                keyValue!=questioner_node_to_outerIndex_and_innerIndex.end();
                keyValue++)
            {
                std::uint64_t questioner = keyValue->first;
                std::uint64_t outerIndex = keyValue->second.first;
                assert(outerIndex>=0);
                assert(outerIndex<nodes_to_ask_question.size());
                assert(outerIndex<nodes_that_ask_the_question.size());
                assert(outerIndex<question_parameters.size());
                std::uint64_t innerIndex = keyValue->second.second;
                assert(innerIndex>=0);
                assert(innerIndex<nodes_to_ask_question[outerIndex].size());
                assert(innerIndex<nodes_that_ask_the_question[outerIndex].size());
                assert(questioner==nodes_that_ask_the_question[outerIndex][innerIndex]);
                assert(innerIndex<question_parameters[outerIndex].size());
            }
            structureStatus = PrepareQuestionsToSend;
        };
        void finalizeAddingQuestionsToSend()
        {
            int number_ranks = MPIWrapper::get_number_ranks();
            assert(structureStatus==PrepareQuestionsToSend);
            assert(nodes_to_ask_question.size()==nodes_that_ask_the_question.size());
            assert(nodes_that_ask_the_question.size()==question_parameters.size());
            
            list_index_to_adressee_rank.resize(nodes_to_ask_question.size(),-1);
            for(auto iter=rank_to_outerIndex.begin(); iter!=rank_to_outerIndex.end(); iter++)
            {
                assert(iter->first<number_ranks && iter->first>=0);
                assert(iter->second<nodes_to_ask_question.size() && iter->second>=0);
                assert(list_index_to_adressee_rank[iter->second]==-1);
                list_index_to_adressee_rank[iter->second] = iter->first;
            }

            
            addressee_ranks_to_nbrOfQuestions.resize(number_ranks,0);
            for(int index=0;index<list_index_to_adressee_rank.size();index++)
            {
                std::uint64_t rank = list_index_to_adressee_rank[index];
                assert(rank<number_ranks);
                addressee_ranks_to_nbrOfQuestions[rank] = nodes_to_ask_question[index].size();
            }

            
            structureStatus = ClosedQuestionsPreparation;
            
            for(auto keyValue = questioner_node_to_outerIndex_and_innerIndex.begin();
                keyValue!=questioner_node_to_outerIndex_and_innerIndex.end();
                keyValue++)
            {
                std::uint64_t questioner = keyValue->first;
                std::uint64_t outerIndex = keyValue->second.first;
                assert(outerIndex>=0);
                assert(outerIndex<nodes_to_ask_question.size());
                assert(outerIndex<list_index_to_adressee_rank.size());
                assert(outerIndex<nodes_that_ask_the_question.size());
                assert(outerIndex<question_parameters.size());
                std::uint64_t innerIndex = keyValue->second.second;
                assert(innerIndex>=0);
                assert(innerIndex<nodes_to_ask_question[outerIndex].size());
                assert(innerIndex<nodes_that_ask_the_question[outerIndex].size());
                assert(questioner==nodes_that_ask_the_question[outerIndex][innerIndex]);
                assert(innerIndex<question_parameters[outerIndex].size());
            }
        };

    //Funktions for use on answerer rank side
        void setQuestionsReceived
        (
            std::vector<std::uint64_t>& total_nodes_to_ask_question,
            std::vector<Q_parameter>& total_question_parameters,
            std::vector<int>& rank_size,
            std::vector<int>& rank_displ
        )
        {
            assert(rank_size.size()==rank_displ.size());
            for(int rank=0;rank<rank_size.size();rank++)
            {
                int nbr_of_questions = rank_size[rank];
                if(nbr_of_questions > 0)
                {
                    int outerIndex = list_index_to_adressee_rank.size();
                    rank_to_outerIndex.insert(std::pair<std::uint64_t,std::uint64_t>(rank,outerIndex));
                    
                    assert(outerIndex==nodes_to_ask_question.size());
                    assert(rank_displ[rank]<total_nodes_to_ask_question.size() && rank_displ[rank]>=0);
                    
                    list_index_to_adressee_rank.push_back(rank);
                    nodes_to_ask_question.push_back({});
                    nodes_to_ask_question.back().resize(nbr_of_questions);
                    assert(nodes_to_ask_question.back().size()==nbr_of_questions);
                    assert(rank_displ[rank]>=0 && rank_displ[rank]<total_nodes_to_ask_question.size());
                    std::memcpy(nodes_to_ask_question.back().data(),&total_nodes_to_ask_question[rank_displ[rank]],                        nbr_of_questions*sizeof(std::uint64_t));
                    
                    question_parameters.push_back({});
                    question_parameters.back().resize(nbr_of_questions);
                    assert(outerIndex<question_parameters.size());
                    assert(rank_displ[rank]<total_question_parameters.size() && rank_displ[rank]>=0);
                    assert(nbr_of_questions<=question_parameters[outerIndex].size());
                    std::memcpy(question_parameters.back().data(),&total_question_parameters[rank_displ[rank]],nbr_of_questions*sizeof(Q_parameter));
                    
                    addressee_ranks_to_nbrOfQuestions.push_back(nbr_of_questions);
                }
            }
            //throw std::string("1544");
        };
        void computeAnswersToQuestions
        (
            const DistributedGraph& dg,
            std::function<A_parameter(const DistributedGraph& dg,std::uint64_t node_local_ind,Q_parameter para)> generateAnswers
        )
        {
            assert(nodes_to_ask_question.size()==question_parameters.size());
            answers_to_questions.resize(question_parameters.size());
            for(int i=0; i<question_parameters.size(); i++)
            {
                assert(nodes_to_ask_question[i].size()==question_parameters[i].size());
                answers_to_questions[i].resize(question_parameters[i].size());
                for(int j=0; j<question_parameters[i].size(); j++)
                {
                    std::uint64_t node_local_ind = nodes_to_ask_question[i][j];
                    Q_parameter para = question_parameters[i][j];
                    answers_to_questions[i][j] = generateAnswers(dg,node_local_ind,para);
                }            
                //std::cout<<"Rank:"<<MPIWrapper::get_my_rank()<<"  "<<i<<"/"<<question_parameters.size()<<"  Generated answer for:"<<answers_to_questions[i].size()<<std::endl; 
            }
        };
        void setAnswers
        (
            std::vector<A_parameter>& total_answers,
            std::vector<int>& rank_size,
            std::vector<int>& rank_displ
        )
        {
            int number_answer_blocks = 0;
            for(int rank=0;rank<rank_size.size();rank++)
            {
                if(rank_size[rank]>0)
                {
                    number_answer_blocks++;
                }
            }
            assert(number_answer_blocks==nodes_to_ask_question.size());
            assert(number_answer_blocks==list_index_to_adressee_rank.size());
            assert(number_answer_blocks==nodes_that_ask_the_question.size());
            assert(number_answer_blocks==question_parameters.size());
            
            answers_to_questions.resize(number_answer_blocks);
            for(int rank=0;rank<rank_size.size();rank++)
            {
                int nbr_of_answers = rank_size[rank];
                if(nbr_of_answers > 0)
                {
                    auto keyValue = rank_to_outerIndex.find(rank);
                    assert(keyValue!=rank_to_outerIndex.end());
                    int outerIndex = keyValue->second;
                    assert(outerIndex<answers_to_questions.size() && outerIndex>=0);
                    assert(answers_to_questions[outerIndex].size()==0);
                    assert(nbr_of_answers==nodes_to_ask_question[outerIndex].size());
                    assert(nbr_of_answers==nodes_that_ask_the_question[outerIndex].size());
                    assert(nbr_of_answers==question_parameters[outerIndex].size());            answers_to_questions[outerIndex].resize(nbr_of_answers);
                    
                    std::memcpy(answers_to_questions[outerIndex].data(),&total_answers[rank_displ[rank]],nbr_of_answers*sizeof(A_parameter));
                }
            }
        };
        std::unique_ptr<std::vector<A_parameter>> getAnswersOfQuestionerNode
        (
            std::uint64_t node_local_ind
        )
        {
            auto answers = std::make_unique<std::vector<A_parameter>>();
            for(auto keyValue = questioner_node_to_outerIndex_and_innerIndex.find(node_local_ind);
                keyValue!=questioner_node_to_outerIndex_and_innerIndex.end() && keyValue->first == node_local_ind;
                keyValue++)
            {
                
                std::uint64_t outerIndex = keyValue->second.first;
                assert(outerIndex<answers_to_questions.size() && outerIndex>=0);
                std::uint64_t innerIndex = keyValue->second.second;
                if(!(innerIndex<answers_to_questions[outerIndex].size() && innerIndex>=0))
                    std::cout<<"-------"<<outerIndex<<"---------"<<innerIndex<<"---------"<<answers_to_questions[outerIndex].size()<<std::endl;
                assert(innerIndex<answers_to_questions[outerIndex].size() && innerIndex>=0);
                answers->push_back(answers_to_questions[outerIndex][innerIndex]);
            }
            return std::move(answers);
        };
        std::vector<int>& get_adressee_ranks_to_nbrOfQuestions()
        {
            assert(structureStatus==ClosedQuestionsPreparation);
            return addressee_ranks_to_nbrOfQuestions;
        };
        std::vector<std::uint64_t>& get_nodes_to_ask_question_for_rank(std::uint64_t rank)
        {
            auto keyValue = rank_to_outerIndex.find(rank);
            if(keyValue!=rank_to_outerIndex.end() && nodes_to_ask_question.size()!=0)
            {
                std::uint64_t outerIndex = keyValue->second;
                assert(outerIndex<nodes_to_ask_question.size());
                return nodes_to_ask_question[outerIndex];
            }
            return dummy_nodes_to_ask_question;
        };
        std::vector<Q_parameter>& get_question_parameters_for_rank(std::uint64_t rank)
        {
            auto keyValue = rank_to_outerIndex.find(rank);
            if(keyValue!=rank_to_outerIndex.end() && question_parameters.size()!=0)
            {
                std::uint64_t outerIndex = keyValue->second;
                assert(outerIndex<question_parameters.size());
                return question_parameters[outerIndex];
            }
            return dummy_question_parameters;
        };
        std::vector<A_parameter>& get_answers_for_rank(std::uint64_t ranks)
        {
            auto keyValue = rank_to_outerIndex.find(ranks);
            if(keyValue!=rank_to_outerIndex.end() && answers_to_questions.size()!=0)
            {
                std::uint64_t outerIndex = keyValue->second;
                assert(outerIndex<answers_to_questions.size());
                return answers_to_questions[outerIndex];
            }
            else
            {
                return dummy_answers_to_questions;
            }
        };
        
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
};
