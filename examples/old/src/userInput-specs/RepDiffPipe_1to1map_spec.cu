/**
  * Input spec file for MERCATOR.
  * Specifies a version of the DiffTypePipe pipeline with the
  *  pipeline replicated after the Begin node.
  * 
  * Each node in the main pipeline is (trivially) of a different
  *  type, so that they cannot be executed together.  
  *
  * Intention: supply same code for each main-pipeline node,
  *  then compare performance to that of a similar pipeline with
  *  main pipeline nodes being of the same type so that they can
  *  be executed together.  This should allow for measuring the
  *  benefit of the Function-Centric model, i.e. combined execution
  *  of nodes with same code.
  * 
  * Topology: 
  *    Source -> Begin -> A1 -> B1 -> C1 -> D1 -> E1 -> End1 -> Sink1
  *                    -> A2 -> B2 -> C2 -> D2 -> E2 -> End2 -> Sink2
  *
  * Roles:
  *           Begin: setup node
  *           A-J: main processing nodes
  *           End: finish nodes
  *
  * Mapping: 1:1 from elements to threads; i.e., 1 thread
  *          is assigned to each element
  */

// placeholder
struct PipeEltT {
};

/*** App name. ***/
#pragma mtr application RepDiffPipe_1to1map

/*** Module (i.e., module type) specs. ***/

// Setup
#pragma mtr module Begin (PipeEltT[128] -> accept1<PipeEltT>:?1, accept2<PipeEltT> | 1 : 1) 

#pragma mtr module A1 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module B1 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module C1 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module D1 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module E1 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module End1 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 
#pragma mtr module A2 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module B2 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module C2 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module D2 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module E2 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module End2 (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 


// SOURCE Module
#pragma mtr module SOURCE<PipeEltT>

// SINK Module
#pragma mtr module SINK<PipeEltT>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node BeginNode : Begin
//rep 1
#pragma mtr node A1node<NodeDataT> : A1
#pragma mtr node B1node<NodeDataT> : B1
#pragma mtr node C1node<NodeDataT> : C1
#pragma mtr node D1node<NodeDataT> : D1
#pragma mtr node E1node<NodeDataT> : E1
#pragma mtr node EndNode1 : End1
#pragma mtr node sinkNodeAccept1 : SINK<PipeEltT>
//rep 2
#pragma mtr node A2node<NodeDataT> : A2
#pragma mtr node B2node<NodeDataT> : B2
#pragma mtr node C2node<NodeDataT> : C2
#pragma mtr node D2node<NodeDataT> : D2
#pragma mtr node E2node<NodeDataT> : E2
#pragma mtr node EndNode2 : End2
#pragma mtr node sinkNodeAccept2 : SINK<PipeEltT>


/*** Edge specs. ***/

// SOURCE -> Begin
#pragma mtr edge sourceNode::outStream -> BeginNode

// Begin -> A1
#pragma mtr edge BeginNode::accept1 -> A1node

// A1-> B1
#pragma mtr edge A1node::accept -> B1node

// B1 -> C1
#pragma mtr edge B1node::accept -> C1node

// C1 -> D1
#pragma mtr edge C1node::accept -> D1node

// D1 -> E1
#pragma mtr edge D1node::accept -> E1node

// E1 -> End1
#pragma mtr edge E1node::accept -> EndNode1

// End1 -> SINK1
#pragma mtr edge EndNode1::accept -> sinkNodeAccept1

// Begin -> A2
#pragma mtr edge BeginNode::accept2 -> A2node

// A2-> B2
#pragma mtr edge A2node::accept -> B2node

// B2 -> C2
#pragma mtr edge B2node::accept -> C2node

// C2 -> D2
#pragma mtr edge C2node::accept -> D2node

// D2 -> E2
#pragma mtr edge D2node::accept -> E2node

// E2 -> End2
#pragma mtr edge E2node::accept -> EndNode2

// End2 -> SINK1
#pragma mtr edge EndNode2::accept -> sinkNodeAccept2
