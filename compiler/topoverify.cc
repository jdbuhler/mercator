//
// TOPOVERIFY.CC
// Verify that a MERCATOR application has legal topology, and
// compute some properties of that topology.
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>

#include "topoverify.h"

using namespace std;

/*
 * THEORY
 *
 * We accept only application topologies with no nested or overlapping
 * cycles, since these are the only topologies for which we know how
 * to prevent deadlock.  More specifically, we enforce the following
 * property, which we refer to as a "legal app":
 *
 *   No node of the app graph may be part of more than one directed cycle.
 *
 * To check this property, we perform DFS starting from the app's
 * unique source node.

 * Claim: an app is legal iff no node v explored by DFS has two
 * outgoing edges that both terminate on ancestors of v in the DFS
 * tree.
 *
 * Pf: Suppose an app A has a node v with outgoing edges e and e',
 * such that DFS encounters ancestors x and x' of v when searching via
 * e and e', respectively.  Then there are two cycles v ---> x --> v
 * and v ---> x' ---> v, which are distinct because e and e' point to
 * different nodes.  Hence A is not legal.
 *
 * Conversely, suppose A is not legal.  Then some vertex is part of
 * two directed cycles. Let v be the first such vertex encountered by
 * DFS for which the cycles involve different edges e and e' out of v;
 * there must be some such v, or else the two cycles would not be
 * distinct.  Then DFS via both e and e' will encounter ancestors of
 * v.  QED
 *
 * We can check the legality condition by having recursive DFS return
 * the last in-progress vertex it encountered.  If we are searching
 * from vertex v, and two edges e and e' out of v return vertices x
 * and x' that are *still* in-progress, then x and x' must be
 * ancestors of v, and the graph is illegal.
 *
 * A second property enforced for apps is that we may not have two
 * incoming edges to a node, unless one edge is part of a directed
 * cycle involving that node.  If two edges enter node v, neither of
 * which is part of a cycle involving v, then the second such edge
 * found by DFS will find that v is finished.  As a result of this
 * property, we observe that each cycle has a unique head node, which
 * is the only node where data can enter from outside the cycle.
 *
 * A third property enforced for apps is that a cycle cannot amplify
 * its input.  That is, each input to the head cannot generate more
 * than one input via the cycle's back edge.  Otherwise, the queue
 * of the cycle's head might grow without bound.  We check this
 * property by computing for each node the total amplification factor
 * from the source to its input and ensuring that, whenever we encounter
 * a cycle head, its amplification factor via the path that traverses
 * the cycle equals the factor before we entered the cycle.
 *
 * If the graph passes all checks, then when DFS finishes, each
 * node holds its predecessor in the DFS tree, and nodes that
 * are heads of cycles also hold their cycle predecessor.
 */

int TopologyVerifier::currentId = 1;  //Initialize the enumerate ID tagging

void TopologyVerifier::verifyTopology(App *app)
{
  app->refCounts.push_back(0);	//Initialize refCounts vector for later, index 0 is an invalid enumId
  for(unsigned int i = 0; i < app->modules.size(); ++i) {
	app->isPropagate.push_back(0);
  }
  dfsVisit(app->sourceNode, nullptr, 1, 0, app);
  
  for (Node *node : app->nodes)
    {
	cout << "NODE " << node->get_name() << "\tMODULE NAME " << node->get_moduleType()->get_name() << endl;
      if (node->dfsStatus == Node::NotVisited)
	{
	  cerr << "ERROR: node " << node->get_name()
	       << " is not connected to the source node!"
	       << endl;
	  abort();
	}
      
      if (node->get_moduleType()->isSource()) // no input queue
	continue;
      
      // compute max inputs/run call and outputs/input for tree parent
      {
	const ModuleType *usmod  = node->treeEdge->usNode->get_moduleType();
	const Channel *usChannel = node->treeEdge->usChannel;
	
	int maxInputsPerFiring = 
	  usmod->get_inputLimit() * 
	  usmod->get_nElements()/usmod->get_nThreads();
	
	node->queueSize = maxInputsPerFiring * usChannel->maxOutputs;
	cout << "\t\tMAX INPUTS PER FIRING: " << maxInputsPerFiring << "\tUPSTREAM CHANNEL MAX OUTPUTS: " << usChannel->maxOutputs << endl;
      }
      
      // if the node has a cycle parent, add to its queue size and
      // set the reserved slots on its tree parent edge.
      if (node->cycleEdge)
      {
	const ModuleType *usmod  = node->cycleEdge->usNode->get_moduleType();
	const Channel *usChannel = node->cycleEdge->usChannel;
	
	int maxInputsPerFiring = 
	  usmod->get_inputLimit() * 
	  usmod->get_nElements()/usmod->get_nThreads();
	
	int dsReservedSlots = maxInputsPerFiring * usChannel->maxOutputs;
	
	node->treeEdge->dsReservedSlots = dsReservedSlots;
	node->queueSize                += dsReservedSlots;
      }
    }

	//app->refCounts.push_back(std::make_pair(1,1));

    //for(unsigned int i = 0; i < app->refCounts.size(); ++i) {
	//cout << "APP REF COUNT " << i << ": " << app->refCounts.at(i) << endl;
    //}

    // Check to make sure that no enumerates are unclosed,
    // that is, check to see if any Sinks have an enumerateId.
    for(Node* node : app->nodes) {
	if(node->get_moduleType()->get_isSink() && node->get_enumerateId() > 0) {
		cerr << "ERROR: Sink node "
		     << node->get_name()
		     << " has an enumerate ID. "
		     << "Missing an aggregate channel "
		     << "before this node."
		     << endl;
		abort();
	}
    }

    //Debug stuff
    /*
    for(ModuleType* mod : app->modules) {
	cout << "ModuleType: " << mod->get_name() << endl;
	for(Node* node : mod->nodes) {
		cout << "Node: " << node->get_name() << "\tEnumID: " << node->get_enumerateId() << "\tRef Count: " << app->refCounts.at(node->get_enumerateId()) << endl; 
	}
	cout << endl;
    }
    */
}
    

// multiplier: max possible inputs to this node generated by one input
// to the application's source node

Node *TopologyVerifier::dfsVisit(Node *node,
				 Edge *parentEdge,
				 long multiplier,
				 int enumId,
				 App *app)
{
  //cout << "NODE ENUMERATE?: " << (node->get_moduleType()->get_isEnumerate()) << "\tENUMID = " << node->get_enumerateId() << endl;
  
  if (node->dfsStatus == Node::InProgress)
    {
      node->cycleEdge = parentEdge;
      
      if (multiplier != node->multiplier)
	{
	  cerr << "ERROR: cycle with head at node "
	       << node->get_name()
	       << " has an amplification factor of "
	       << multiplier / node->multiplier
	       << " > 1!"
	       << endl;
	  abort();
	}
      
      return node; // return in-progress node 

    }
  else if (node->dfsStatus == Node::Finished)
    {
      cerr << "ERROR: node " << node->get_name()
	   << " has two convergent input edges, " << endl
	   << "  neither of which is a back edge!" 
	   << endl;
      abort();
    }
  else
    {
      node->dfsStatus  = Node::InProgress; 
      
      node->treeEdge   = parentEdge;
      node->multiplier = multiplier;
      
      const ModuleType *mod = node->get_moduleType();

      if(node->get_moduleType()->get_isEnumerate()) 
        {
	  if(enumId != 0)
	    {
	      cerr << "ERROR: node " << node->get_name()
		   << " is nested within another enumerate, " << endl
		   << "  nesting of enumerates currently unsupported!"
		   << endl;
	      abort();
	    }
	  node->set_enumerateId(currentId);
	  app->refCounts.push_back(0);
	  currentId += 1;
        }
      else
	{
	  node->set_enumerateId(enumId);
	}

      Node *head = nullptr;
      for (int j = 0; j < mod->get_nChannels(); j++)
	{
	  Edge *e = node->dsEdges[j];
	  if (e == nullptr) // output channel is not connected
	    continue;
	  //cout << "IS CHANNEL AGGREGATE?: " << (e->usChannel->isAggregate) << endl;
	  
	  long nextAmpFactor = e->usChannel->maxOutputs;
	  
	  if(e->usChannel->isAggregate) {
	    //cout << "UPSTREAM CHANNEL IS AGGREGATE" << endl;
	    //node->get_mutableModuleType()->add_refCount(enumId);
	    //add_refCount(enumId);
	    app->refCounts.at(enumId) += 1;
	    node->set_enumerateId(0);
	    //cout << "MOD REF COUNT: " << mod->get_refCount(enumId) << endl;
	  }

	  Node *nextHead = dfsVisit(e->dsNode, 
				    e,
				    multiplier * nextAmpFactor,
				    node->get_enumerateId(),
				    app);
	  
	  if (nextHead && nextHead->dfsStatus == Node::InProgress)
	    {
	      if (head != nullptr)
		{
		  cerr << "ERROR: node " << node->get_name()
		       << " is on two distinct cycles!" 
		       << endl;
		  abort();
		}
	      else
		head = nextHead;
	    }

	}
      
      // Set whther or not the module index needs begin/end stubs
      if((node->get_enumerateId() > 0 && !(node->get_moduleType()->get_isEnumerate())))
	{
	  app->isPropagate.at(mod->get_idx()) = true;
	}

      node->dfsStatus = Node::Finished;
      
      return head;
    }
}
