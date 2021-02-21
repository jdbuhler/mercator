//
// APP.CC
// Internal representation of an application
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <climits>

#include "app.h"

#include "inputspec.h"

#include "options.h"

using namespace std;

DataType::DataType(const input::DataType *ti)
{
  name = ti->name;
  
  if (ti->from)
    from = new DataType(ti->from);
  else
    from = nullptr;
}

DataType::DataType(const string &iname)
{
  name = iname;
  from = nullptr;
}

DataType::DataType(const string &iname, const string &fname)
{
  name = iname;
  from = new DataType(fname);
}

///////////////////////////////////////////


Node::Node(const string &iname,
	   ModuleType *imt,
	   unsigned int imIdx)
  : name(iname),
    moduleType(imt),
    mIdx(imIdx),
    regionId(0),
    enumerateId(0),
    nTerminalNodes(0),
    isSource(false),
    _isTerminalNode(false),
    queueSize(0),
    enumerator(nullptr),
    treeEdge(nullptr),
    cycleEdge(nullptr),
    dfsStatus(Node::NotVisited),
    startTime(0),
    multiplier(0)
{
  unsigned int nChannels = moduleType->get_nChannels();
  
  dsEdges = new Edge * [nChannels];
  for (unsigned int j = 0; j < nChannels; j++)
    dsEdges[j] = nullptr;
}

Node::~Node()
{
  for (unsigned int j = 0; j < moduleType->get_nChannels(); j++)
    {
      if (dsEdges[j])
	delete dsEdges[j];
    }
  
  delete [] dsEdges;
}

//////////////////////////////////////////

ModuleType::ModuleType(const string &iname,
		       unsigned int iidx,
		       DataType *iinputType,
		       unsigned int inChannels,
		       unsigned int iflags,
		       unsigned int iinputsPerThread,
		       unsigned int iinputLimit)
  : name(iname),
    idx(iidx),
    inputType(iinputType),
    nChannels(inChannels),
    flags(iflags),
    inputLimit(iinputLimit),
    nInputsPerThread(iinputsPerThread)
{
  channels = new Channel * [nChannels];
  for (unsigned int j = 0; j < nChannels; j++)
    channels[j] = nullptr;
}


ModuleType::~ModuleType()
{
  for (DataItem *param : moduleParams)
    {
      delete param;
    }

  for (DataItem *param : nodeParams)
    {
      delete param;
    }
  
  for (Node * node : nodes)
    {
      delete node;
    }
  
  for (unsigned int j = 0; j < nChannels; j++)
    {
      if (channels[j])
	delete channels[j];
    }
  delete [] channels;
  
  delete inputType;
}


///////////////////////////////////////////////////////////
// DEBUG PRINTING
///////////////////////////////////////////////////////////

void DataType::print() const
{
  cout << name;
  if (from)
    {
      cout << " FROM ";
      from->print();
    }
}

void DataItem::print() const
{
  cout << name << " : ";
  type->print();
}

void Edge::print() const
{
  cout << usNode->get_name() << "::" 
       << usChannel->name    << " -> " 
       << dsNode->get_name()
       << " (" << dsReservedSlots << ')';
}

void Channel::print() const
{
  cout << name << " : ";
  type->print();
  cout << " [" << maxOutputs << ']';
}

void Node::print() const
{
  cout << "NODE " << name << " : " << moduleType->get_name() << endl;
  cout << " * " << moduleType->get_nChannels() << " outgoing edges" << endl;
  for (unsigned int j = 0; j < moduleType->get_nChannels(); j++)
    {
      cout << "  ";
      if (dsEdges[j])
	dsEdges[j]->print();
      else
	cout << "null" << endl;
      
      cout << endl;
    }

  if (isSource)
    cout << "* Node is APPLICATION SOURCE\n";
  
  cout << " * Tree predecessor: " 
       << (treeEdge ? treeEdge->usNode->name : "NONE")
       << endl;
  
  if (cycleEdge)
    cout << " * Cycle predecessor: " << cycleEdge->usNode->name << endl;
}


void ModuleType::print() const
{
  cout << "MODULE TYPE " << name << " : ";
  inputType->print();
  cout << endl << "  ->  " << endl;
  
  for (unsigned int j = 0; j < nChannels; j++)
    {
      cout << "  ";
      channels[j]->print();
      if (j < nChannels - 1)
	cout << ',';
      cout << endl;
    }
  
  if (nChannels == 0)
    {
      cout << "  NULL";
      if (isSink())
	cout << " [sink]";
      cout << endl;
    }
  
  cout << "      FLAGS: " << flags << endl;
  cout << "     ILIMIT: " << inputLimit << endl;
  cout << "ELTS/THREAD: " << nInputsPerThread << endl;
  
  if (moduleParams.size() > 0)
    {
      cout << "   PER-MODULE PARAMS:" << endl;
      for (const DataItem *p : moduleParams)
	{
	  cout << "    ";
	  p->print();
	  cout << endl;
	}
    }

  if (nodeParams.size() > 0)
    {
      cout << "   PER-NODE PARAMS:" << endl;
      for (const DataItem *p : nodeParams)
	{
	  cout << "    ";
	  p->print();
	  cout << endl;
	}
    }
  
  cout << " NODES OF THIS TYPE:" << endl;
  for (Node *node : nodes)
    {
      node->print();
      cout << "***" << endl;
    }
}


void App::print() const
{
  cout << "APP " << name << endl;
  
  if (params.size() > 0)
    {
      cout << "   PARAMS:" << endl;
      for (const DataItem *p : params)
	{
	  cout << "    ";
	  p->print();
	  cout << endl;
	}
    }

  cout << " THREAD WIDTH: " << threadWidth << endl;
  cout << " SOURCE NODE: " << sourceNode->get_name() << endl;
  
  for (ModuleType *mod : modules)
    {
      mod->print();
      cout << "-----------" << endl;
    }
}
  
