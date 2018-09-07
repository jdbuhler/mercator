/**
  * Input spec file for MERCATOR.
  * Specifies a pipeline consisting of a single main node
  *  plus one setup node and one cleanup (or base-case) node.
  * 
  * The main node  ("uber node") contains all application logic.
  *
  * Intention: Compare performance to that of an app with 
  *  app logic spread over multiple nodes.  This will highlight
  *  tradeoffs between the overhead of MERCATOR data movement
  *  between nodes and the benefit of parallel execution and 
  *  utilization management.
  * 
  * Topology: 
  *    Source -> Begin -> Uber -> End -> Sink
  *
  * Roles:
  *           Begin: setup node
  *           Uber: main processing node
  *           End: finish node
  *
  * Mapping: 1:2 from elements to threads; i.e., 2 threads
  *          are assigned to each element
  */

// placeholder
struct PipeEltT {
};

/*** App name. ***/
#pragma mtr application UberNodePipe_1to2map

/*** Module (i.e., module type) specs. ***/

// Setup
#pragma mtr module Begin (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module Uber (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 2) 

#pragma mtr module End (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 


// SOURCE Module
#pragma mtr module SOURCE<PipeEltT>

// SINK Module
#pragma mtr module SINK<PipeEltT>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node BeginNode : Begin
#pragma mtr node Ubernode<NodeData5FiltT> : Uber
#pragma mtr node EndNode : End
#pragma mtr node sinkNodeAccept : SINK<PipeEltT>


/*** Edge specs. ***/

// SOURCE -> Begin
#pragma mtr edge sourceNode::outStream -> BeginNode

// Begin -> Uber
#pragma mtr edge BeginNode::accept -> Ubernode

// Uber -> End
#pragma mtr edge Ubernode::accept -> EndNode

// End -> SINK
#pragma mtr edge EndNode::accept -> sinkNodeAccept

