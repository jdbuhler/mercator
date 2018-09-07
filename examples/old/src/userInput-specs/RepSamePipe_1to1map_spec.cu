/**
  * Input spec file for MERCATOR.
  * Specifies a version of the SameTypePipe pipeline with the
  *  pipeline replicated after the Begin node.
  * 
  * Intention: Compare performance to that of a similar pipeline with
  *  main pipeline nodes being of (trivially) different types so that
  *  they cannot be executed together. This should allow for measuring
  * the benefit of the Function-Centric model, i.e. combined execution
  *  of nodes with same code.
  * 
  * Topology: 
  *    Source -> Begin -> A11 -> A12 -> A13 -> A14 -> A15 -> End1 -> Sink1
  *                    -> A21 -> A22 -> A23 -> A24 -> A25 -> End2 -> Sink2
  *
  *
  * Roles:
  *           Begin: setup node
  *           A.1-A.5: main processing nodes
  *           End: finish nodes
  *
  * Mapping: 1:1 from elements to threads; i.e., 1 thread
  *          is assigned to each element
  */

// placeholders-- necessary for element types,
//  but not user-defined node-, module-, and app-level parameter data types
struct PipeEltT {
};

/*** App name. ***/
#pragma mtr application RepSamePipe_1to1map

/*** Module (i.e., module type) specs. ***/

// Setup
#pragma mtr module Begin (PipeEltT[128] -> accept1<PipeEltT>:?1, accept2<PipeEltT> | 1 : 1) 
#pragma mtr module A (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module End (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 


// SOURCE Module
#pragma mtr module SOURCE<PipeEltT>

// SINK Module
#pragma mtr module SINK<PipeEltT>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
//#pragma mtr node Anode<MyNodeData> : A
#pragma mtr node BeginNode : Begin

// rep 1
#pragma mtr node A11node<NodeDataT> : A
#pragma mtr node A12node<NodeDataT> : A
#pragma mtr node A13node<NodeDataT> : A
#pragma mtr node A14node<NodeDataT> : A
#pragma mtr node A15node<NodeDataT> : A
#pragma mtr node EndNode1 : End
#pragma mtr node sinkNodeAccept1 : SINK<PipeEltT>

// rep 2
#pragma mtr node A21node<NodeDataT> : A
#pragma mtr node A22node<NodeDataT> : A
#pragma mtr node A23node<NodeDataT> : A
#pragma mtr node A24node<NodeDataT> : A
#pragma mtr node A25node<NodeDataT> : A
#pragma mtr node EndNode2 : End
#pragma mtr node sinkNodeAccept2 : SINK<PipeEltT>


/*** Edge specs. ***/

// SOURCE -> Begin
#pragma mtr edge sourceNode::outStream -> BeginNode

// Begin -> A11
#pragma mtr edge BeginNode::accept1 -> A11node

// A11 -> A12
#pragma mtr edge A11node::accept -> A12node

// A12 -> A13
#pragma mtr edge A12node::accept -> A13node

// A13 -> A14
#pragma mtr edge A13node::accept -> A14node

// A14 -> A15
#pragma mtr edge A14node::accept -> A15node

// A15 -> End1
#pragma mtr edge A15node::accept -> EndNode1

// End1 -> SINK1
#pragma mtr edge EndNode1::accept -> sinkNodeAccept1

// Begin -> A21
#pragma mtr edge BeginNode::accept2 -> A21node

// A21 -> A22
#pragma mtr edge A21node::accept -> A22node

// A22 -> A23
#pragma mtr edge A22node::accept -> A23node

// A23 -> A24
#pragma mtr edge A23node::accept -> A24node

// A24 -> A25
#pragma mtr edge A24node::accept -> A25node

// A25 -> End2
#pragma mtr edge A25node::accept -> EndNode2

// End1 -> SINK1
#pragma mtr edge EndNode2::accept -> sinkNodeAccept2
