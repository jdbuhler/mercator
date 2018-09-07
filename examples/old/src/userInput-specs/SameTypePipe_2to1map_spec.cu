/**
  * Input spec file for MERCATOR.
  * Specifies a pipeline consisting of five nodes, 
  *  plus one setup node and one cleanup (or base-case) node.
  * 
  * Each node in the main pipeline is of the same
  *  type, so that they can be executed together.  
  *
  * Intention: Compare performance to that of a similar pipeline with
  *  main pipeline nodes being of (trivially) different types so that
  *  they cannot be executed together. This should allow for measuring
  * the benefit of the Function-Centric model, i.e. combined execution
  *  of nodes with same code.
  * 
  * Topology: 
  *    Source -> Begin -> A1 -> A2 -> A3 -> A4 -> A5 -> End -> Sink
  *
  * Roles:
  *           Begin: setup node
  *           A1-A5: main processing nodes
  *           End: finish node
  *
  * Mapping: 2:1 from elements to threads; i.e., 2 elements
  *          are assigned to each thread
  */

// placeholders-- necessary for element types,
//  but not user-defined node-, module-, and app-level parameter data types
struct PipeEltT {
};

/*** App name. ***/
#pragma mtr application SameTypePipe_2to1map

/*** Module (i.e., module type) specs. ***/

// Setup
#pragma mtr module Begin (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 
#pragma mtr module A (PipeEltT[256] -> accept<PipeEltT>:?1 | 2 : 1) 

#pragma mtr module End (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 


// SOURCE Module
#pragma mtr module SOURCE<PipeEltT>

// SINK Module
#pragma mtr module SINK<PipeEltT>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
//#pragma mtr node Anode<MyNodeData> : A
#pragma mtr node BeginNode : Begin
#pragma mtr node A1node<NodeDataT> : A
#pragma mtr node A2node<NodeDataT> : A
#pragma mtr node A3node<NodeDataT> : A
#pragma mtr node A4node<NodeDataT> : A
#pragma mtr node A5node<NodeDataT> : A
#pragma mtr node EndNode : End
#pragma mtr node sinkNodeAccept : SINK<PipeEltT>


/*** Edge specs. ***/

// SOURCE -> Begin
#pragma mtr edge sourceNode::outStream -> BeginNode

// Begin -> A1
#pragma mtr edge BeginNode::accept -> A1node

// A1 -> A2
#pragma mtr edge A1node::accept -> A2node

// A2 -> A3
#pragma mtr edge A2node::accept -> A3node

// A3 -> A4
#pragma mtr edge A3node::accept -> A4node

// A4 -> A5
#pragma mtr edge A4node::accept -> A5node

// A5 -> End
#pragma mtr edge A5node::accept -> EndNode


// End -> SINK
#pragma mtr edge EndNode::accept -> sinkNodeAccept

