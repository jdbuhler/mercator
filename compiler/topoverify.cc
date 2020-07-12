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

void TopologyVerifier::verifyTopology(App *app)
{
  refCount.clear();
  refCount.push_back(0);     // reserve entry for global region 0
  
  parentRegion.clear();
  parentRegion.push_back(0); // reserve entry for global region 0
  
  nextRegionId = 1;          // ID of first non-global region
  
  dfsVisit(app->sourceNode, nullptr, 1, 0, app);
  
  for (Node *node : app->nodes)
    {
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
      
      // record reference count for enumerate nodes, and make sure
      // that from type did not leak through to sink
      if (node->enumerateId > 0)
	{
	  if (node->moduleType->get_isSink())
	    {
	      cerr << "ERROR: Sink node "
		   << node->get_name()
		   << " has an enumerate ID. "
		   << "Missing an aggregate channel "
		   << "before this node."
		   << endl;
	      abort();
	    }
	  
	  node->refCount = refCount[node->enumerateId];
	}
    }
}
    

// multiplier: max possible inputs to this node generated by one input
// to the application's source node

Node *TopologyVerifier::dfsVisit(Node *node,
				 Edge *parentEdge,
				 long multiplier,
				 int regionId,
				 App *app)
{
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
      
      node->regionId = regionId;
      
      if (node->moduleType->get_isEnumerate())
	{
	  // New enumerate node gets a new, distinct enum ID greater
	  // than that of the region that contains it, which becomes
	  // the region ID for its children.
	  
	  node->enumerateId = nextRegionId++;
	  parentRegion.push_back(regionId); // remember parent of new region
	  regionId = node->enumerateId;     // set new region for children
	  
	  refCount.push_back(0);   // reserve space for region refcount
	  
	  cout << "FOUND ENUMERATE:" << endl
	       << "\t" << node->moduleType->get_name() << endl
	       << "\tRegionID:\t" << node->regionId << endl
	       << "\tEnumerateID:\t" << node->enumerateId << endl;
	}
      
      Node *head = nullptr;
      for (int j = 0; j < mod->get_nChannels(); j++)
	{
	  Edge *e = node->dsEdges[j];
	  if (e == nullptr) // output channel is not connected
	    continue;
	  
	  long nextAmpFactor = e->usChannel->maxOutputs;
	  
	  if (e->usChannel->isAggregate) 
	    {
	      // we are passing out of current region; revert to its parent
	      regionId = parentRegion[regionId];
	      
	      // add a reference count for each edge leaving the region
	      refCount[regionId]++;
	    }
	  
	  Node *nextHead = dfsVisit(e->dsNode, 
				    e,
				    multiplier * nextAmpFactor,
				    regionId,
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
      
      node->dfsStatus = Node::Finished;
      
      return head;
    }
}
