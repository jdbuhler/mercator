#include <iostream>
#include <vector>
#include <unordered_map>

#include "validate.cuh"

using namespace std;

bool validate_lowpassFilterApp_outputs_inorder(const PipeEltT* inputs,
					       int inSize,
					       const PipeEltT* outputs,
					       int outSize,
					       int filterThresh,
					       int gamma)
{
  bool match = true;
  
  vector<int> goldenOut;

  for (int i=0; i < inSize; ++i)
    {
      int ID = inputs[i].get_ID();
      if (ID >= 0 && ID < filterThresh)   // valid item
	{
	  goldenOut.push_back(ID);
	}
    }
  
  if ((unsigned int) outSize != goldenOut.size())
    {
      cout << "ERROR: observed number of outputs "
	   << outSize
	   << " doesn't match expected "
	   << goldenOut.size()
	   << endl;
      match = false;
    }
  else
    cout << "Checking " << outSize << " outputs... " << endl;
  
  for (unsigned int i=0; i < min((size_t) outSize, goldenOut.size()); ++i)
    {
      if (outputs[i].get_ID() != goldenOut[i])
	{
	  cout << "ERROR: output " << i << " with ID "
	       << outputs[i].get_ID()
	       << " does not match expected ID "
	       << goldenOut[i]
	       << endl;
	  match = false;
	}
    }
  
  return match;
}


bool validate_lowpassFilterApp_outputs(const PipeEltT* inputs,
				       int inSize,
				       const PipeEltT* outputs,
				       int outSize,
				       int filterThresh,
				       int gamma)
{
  typedef unordered_map<int, int> IDMap;
  
  bool allValid = true; // innocent until proven guilty
  
  // tally counts of all inbuffer items in hash map
  IDMap inputsMap;
  
  for (int i=0; i < inSize; ++i)
    {
      int ID = inputs[i].get_ID();
      if (ID >= 0 && ID < filterThresh)   // valid item
	{
	  auto record = inputsMap.find(ID);
	  if (record != inputsMap.end()) // exists already
	    record->second++;
	  else    // doesn't exist-- need to insert
	    inputsMap.insert(pair<int, int>(ID, 1));
	}
    }
  
  // tally counts of all outbuffer items in hash map
  IDMap outputsMap;
  
  for (int i = 0; i < outSize; ++i)
    {
      int ID = outputs[i].get_ID();
      if (ID >= 0 && ID < filterThresh)   // valid item
	{
	  auto record = outputsMap.find(ID);
	  if(record != outputsMap.end()) // exists already
	    record->second++;
	  else    // doesn't exist-- need to insert
	      outputsMap.insert(pair<int, int>(ID, 1));
	}
      else  // invalid item
	  {
	    cout << "**OUTPUT ERROR: last filter threshold "
		 << filterThresh
		 << " but output buffer item " << i
		 << " has ID " << ID
		 << endl;
	    allValid = false;
	  }
    }
  
  // compare maps
  
  // scan through output map
  for (auto &mapElt : outputsMap)
    {
      // get current element's ID
      int ID = mapElt.first;
      
      // search for ID in input map and compare num occurrences
      auto inMapIt = inputsMap.find(ID);
      if (inMapIt == inputsMap.end())
	{
	  // doesn't exist
	  cout << "**OUTPUT ERROR: last filter threshold "
	       << filterThresh
	       << ", output buffer item with ID " << ID
	       << " and occurrence count " << mapElt.second
		 << " has no match in input buffer"
	       << endl;
	  allValid = false;
	}
      else if (mapElt.second != inMapIt->second * gamma)
	{
	  // wrong occurrence count
	  cout << "**OUTPUT ERROR: last filter threshold "
	       << filterThresh
	       << ", output buffer item with ID " << ID
	       << " and occurrence count " << mapElt.second
	       << " has match in input buffer, but with count "
	       << inMapIt->second
	       << " instead of expected "
	       << mapElt.second/gamma
	       << endl;
	  allValid = false;
	}
      
      // if we pass above conditionals, output item was found in input map 
      //   with correct occurrence count
    }
  
  // scan through input map; report any items not present in output map
  for (auto &inMapElt : inputsMap)
    {
      // get current element's ID
      int ID = inMapElt.first;
      
      // search for ID in input map and compare num occurrences
      auto outMapIt = outputsMap.find(ID);
      if (outMapIt == outputsMap.end())
	{
	  // doesn't exist
	  cout << "**OUTPUT ERROR: last filter threshold "
	       << filterThresh
	       << ", input buffer item with ID " << ID
	       << " and occurrence count " << inMapElt.second
	       << " has no match in output buffer"
	       << endl;
	  allValid = false;
	}
    }
  
  return allValid;
}
