#include "AlgorithmTests.h"

void test_algorithm_parallelization
(
	std::filesystem::path input_directory
)
{
	const int my_rank = MPIWrapper::get_my_rank();
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();
	std::string testResult;
	
	
// Test AreaConnectivity algorithm parallelization
	std::unique_ptr<AreaConnectivity::AreaConnecMap> areaConnectParallel;
	std::unique_ptr<AreaConnectivity::AreaConnecMap> areaConnectSingleProc_Helge;
	std::unique_ptr<AreaConnectivity::AreaConnecMap> areaConnectSingleProc;
	try{
		areaConnectParallel = AreaConnectivity::compute_area_connectivity_strength(dg); // No runtime errors
		MPIWrapper::barrier();
		areaConnectSingleProc_Helge = AreaConnectivity::areaConnectivityStrengthSingleProc_Helge(dg); // Runtime errors
		MPIWrapper::barrier();
		areaConnectSingleProc = AreaConnectivity::areaConnectivityStrengthSingleProc(dg);
		MPIWrapper::barrier();
		compareAreaConnecMap(*areaConnectParallel,*areaConnectSingleProc);
		compareAreaConnecMap(*areaConnectParallel,*areaConnectSingleProc_Helge);
		compareAreaConnecMap(*areaConnectSingleProc_Helge,*areaConnectSingleProc);
		
		testResult = "AreaConnectivity test completed";
	}
	catch(std::string errorCode)
	{
		testResult = "AreaConnectivity Error :"+errorCode;
	}
	if(my_rank==0)
		std::cout<<testResult<<std::endl;
	
// Test Histogram algorithm parallelization
	std::unique_ptr<Histogram::HistogramData> histogramCountBins;
	std::unique_ptr<Histogram::HistogramData> histogramWidthBins;
	std::unique_ptr<Histogram::HistogramData> histogramCountBins_SingleProc;
	std::unique_ptr<Histogram::HistogramData> histogramWidthBins_SingleProc;
	double bin_width = 1;
	std::uint64_t bin_count = 50;
	try{
		
		histogramCountBins = Histogram::compute_edgeLength_Histogramm_constBinCount(dg,bin_count);
		MPIWrapper::barrier();
		histogramWidthBins = Histogram::compute_edgeLength_Histogramm_constBinWidth(dg,bin_width);
		MPIWrapper::barrier();
		histogramCountBins_SingleProc = Histogram::edgeLengthHistogramm_constBinCountSingleProc(dg,bin_count);
		MPIWrapper::barrier();
		histogramWidthBins_SingleProc = Histogram::edgeLengthHistogramm_constBinWidthSingleProc(dg,bin_width);
		MPIWrapper::barrier();
		compareEdgeLengthHistogram(*histogramCountBins,*histogramCountBins_SingleProc,1e-8);
		compareEdgeLengthHistogram(*histogramWidthBins,*histogramWidthBins_SingleProc,1e-8);
		
		testResult = "Histogram test completed";
	}
	catch(std::string errorCode)
	{
		testResult = "Histogram Error :"+errorCode;
	}
	if(my_rank==0)
		std::cout<<testResult<<std::endl;

// Test Modularity algorithm parallelization
	double modularityPar,modularitySeq;
	try{
		modularityPar = Modularity::compute_modularity(dg);
		modularitySeq = Modularity::computeModularitySingleProc(dg);
		double absoluteError = std::abs(modularityPar-modularitySeq);
		double relativeError = absoluteError / 0.5*(modularityPar+modularitySeq);
		if(relativeError>1e-8)
		{
			std::stringstream errorCode;
			errorCode<<"modularityPar:"<<modularityPar<<"   modularitySeq:"<<modularitySeq<<"    absoluteError:"<<absoluteError<<"   relativeError:"<<relativeError;
			throw errorCode.str();
		}
		
		testResult = "Modularity test completed";
	}
	catch(std::string errorCode)
	{
		testResult = "Modularity Error :"+errorCode;
	}
	if(my_rank==0)
		std::cout<<testResult<<std::endl;
}

void compareAreaConnecMap(const AreaConnectivity::AreaConnecMap& mapPar,const AreaConnectivity::AreaConnecMap& mapSeq)
{
	for(auto keyValue=mapPar.begin();keyValue!=mapPar.end();keyValue++)
	{
		auto otherKeyValue = mapSeq.find(keyValue->first);
		if(otherKeyValue!=mapSeq.end())
		{
			if(otherKeyValue->second!=keyValue->second)
			{
				std::stringstream errorCode;
				errorCode<<"keyValue:"<<keyValue->first.first<<" --> "<<keyValue->first.second<<"  mapPar:"<<keyValue->second<<"  mapSeq:"<<otherKeyValue->second;
				throw errorCode.str();
			}
		}
		else
		{
			std::stringstream errorCode;
			errorCode<<"keyValue:"<<keyValue->first.first<<" --> "<<keyValue->first.second<<"  does not exist in mapSeq";
			throw errorCode.str();
		}
	}
	for(auto keyValue=mapSeq.begin();keyValue!=mapSeq.end();keyValue++)
	{
		auto otherKeyValue = mapPar.find(keyValue->first);
		if(otherKeyValue!=mapPar.end())
		{
			if(otherKeyValue->second!=keyValue->second)
			{
				std::stringstream errorCode;
				errorCode<<"keyValue:"<<keyValue->first.first<<" --> "<<keyValue->first.second<<"  mapSeq:"<<keyValue->second<<"  mapPar:"<<otherKeyValue->second;
				throw errorCode.str();
			}
		}
		else
		{
			std::stringstream errorCode;
			errorCode<<"keyValue:"<<keyValue->first.first<<" --> "<<keyValue->first.second<<"  does not exist in mapPar";
			throw errorCode.str();
		}
	}
	
	if(mapPar.size()!=mapSeq.size())
	{
		std::stringstream errorCode;
		errorCode<<"mapPar:"<<mapPar.size()<<" || "<<"mapSeq:"<<mapSeq.size();
		throw errorCode.str();
	}
}

void compareEdgeLengthHistogram(const Histogram::HistogramData& histogramPar, const Histogram::HistogramData& histogramSeq, const double epsilon)
{
	const auto my_rank = MPIWrapper::get_my_rank();
	if(my_rank!=0)
		return;

	if(histogramPar.size() != histogramSeq.size())
	{
		std::stringstream errorCode;
		errorCode<<"histogramPar.size():"<<histogramPar.size()<<"  histogramPar.size()"<<histogramPar.size();
		throw errorCode.str();
	}
	
	std::uint64_t total_Edges_Par = 0;
	for(auto entry: histogramPar)
	{
		total_Edges_Par += entry.second;
	}
	
	std::uint64_t total_Edges_Seq = 0;
	for(auto entry: histogramSeq)
	{
		total_Edges_Seq += entry.second;
	}
	
	if(total_Edges_Par != total_Edges_Seq)
	{
		std::stringstream errorCode;
		errorCode<<"total_Edges_Par:"<<total_Edges_Par<<"  total_Edges_Seq:"<<total_Edges_Seq;
		throw errorCode.str();
	}		

	for(int bin=0; bin<histogramPar.size(); bin++)
	{
		auto elemPar = histogramPar[bin];
		auto elemSeq = histogramSeq[bin];
		if(fabs(elemPar.first.first - elemSeq.first.first) > epsilon)
		{
			std::stringstream errorCode;
			errorCode<<"Histograms have different bin boundings in bin:"<<bin<<"  elemPar->first.first:"<<elemPar.first.first<<"   elemSeq->first.first:"<<elemSeq.first.first;
			throw errorCode.str();
		}
		if(fabs(elemPar.first.second - elemSeq.first.second) > epsilon)
		{
			std::stringstream errorCode;
			errorCode<<"Histograms have different bin boundings in bin:"<<bin<<"  elemPar->first.second:"<<elemPar.first.second<<"   elemSeq->first.second:"<<elemSeq.first.second;
			throw errorCode.str();
		}
		if(elemPar.second != elemSeq.second)
		{
			std::stringstream errorCode;
			errorCode<<"Histograms have different bin values in bin:"<<bin<<"  elemPar.second:"<<elemPar.second<<"   elemSeq.second:"<<elemSeq.second;
			throw errorCode.str();
		}
	}
}
