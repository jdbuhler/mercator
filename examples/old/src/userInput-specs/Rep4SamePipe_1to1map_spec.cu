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
  *                    -> A31 -> A32 -> A33 -> A34 -> A35 -> End3 -> Sink3
  *                    -> A41 -> A42 -> A43 -> A44 -> A45 -> End4 -> Sink4
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
#pragma mtr application Rep4SamePipe_1to1map

/*** Module (i.e., module type) specs. ***/

// Setup
#pragma mtr module Begin (PipeEltT[128] -> accept1<PipeEltT>:?1, accept2<PipeEltT>, accept3<PipeEltT>, accept4<PipeEltT> | 1 : 1) 

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

// rep 3
#pragma mtr node A31node<NodeDataT> : A
#pragma mtr node A32node<NodeDataT> : A
#pragma mtr node A33node<NodeDataT> : A
#pragma mtr node A34node<NodeDataT> : A
#pragma mtr node A35node<NodeDataT> : A
#pragma mtr node EndNode3 : End
#pragma mtr node sinkNodeAccept3 : SINK<PipeEltT>

// rep 4
#pragma mtr node A41node<NodeDataT> : A
#pragma mtr node A42node<NodeDataT> : A
#pragma mtr node A43node<NodeDataT> : A
#pragma mtr node A44node<NodeDataT> : A
#pragma mtr node A45node<NodeDataT> : A
#pragma mtr node EndNode4 : End
#pragma mtr node sinkNodeAccept4 : SINK<PipeEltT>
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

// End2 -> SINK2
#pragma mtr edge EndNode2::accept -> sinkNodeAccept2

// Begin -> A31
#pragma mtr edge BeginNode::accept3 -> A31node

// A31 -> A32
#pragma mtr edge A31node::accept -> A32node

// A32 -> A33
#pragma mtr edge A32node::accept -> A33node

// A33 -> A34
#pragma mtr edge A33node::accept -> A34node

// A34 -> A35
#pragma mtr edge A34node::accept -> A35node

// A35 -> End3
#pragma mtr edge A35node::accept -> EndNode3

// End3 -> SINK3
#pragma mtr edge EndNode3::accept -> sinkNodeAccept3

// Begin -> A44
#pragma mtr edge BeginNode::accept4 -> A41node

// A41 -> A42
#pragma mtr edge A41node::accept -> A42node

// A42 -> A43
#pragma mtr edge A42node::accept -> A43node

// A43 -> A44
#pragma mtr edge A43node::accept -> A44node

// A44 -> A45
#pragma mtr edge A44node::accept -> A45node

// A45 -> End4
#pragma mtr edge A45node::accept -> EndNode4

// End4 -> SINK4
#pragma mtr edge EndNode4::accept -> sinkNodeAccept4
