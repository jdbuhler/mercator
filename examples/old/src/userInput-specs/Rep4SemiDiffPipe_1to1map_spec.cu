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
  *                    -> A3 -> B3 -> C3 -> D3 -> E3 -> End2 -> Sink2
  *                    -> A4 -> B4 -> C4 -> D4 -> E4 -> End2 -> Sink2
  *
  * Roles:
  *           Begin: setup node
  *           A-E: main processing nodes, each of own type
  *           End: finish nodes
  *
  * Mapping: 1:1 from elements to threads; i.e., 1 thread
  *          is assigned to each element
  */

// placeholder
struct PipeEltT {
};

/*** App name. ***/
#pragma mtr application Rep4SemiDiffPipe_1to1map

/*** Module (i.e., module type) specs. ***/

// Setup
#pragma mtr module Begin (PipeEltT[128] -> accept1<PipeEltT>:?1, accept2<PipeEltT>, accept3<PipeEltT>, accept4<PipeEltT> | 1 : 1) 

#pragma mtr module A (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module B (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module C (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module D (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module E (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module End (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<PipeEltT>

// SINK Module
#pragma mtr module SINK<PipeEltT>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node BeginNode : Begin
//rep 1
#pragma mtr node A1node<NodeDataT> : A
#pragma mtr node B1node<NodeDataT> : B
#pragma mtr node C1node<NodeDataT> : C
#pragma mtr node D1node<NodeDataT> : D
#pragma mtr node E1node<NodeDataT> : E
#pragma mtr node EndNode1 : End
#pragma mtr node sinkNodeAccept1 : SINK<PipeEltT>
//rep 2
#pragma mtr node A2node<NodeDataT> : A
#pragma mtr node B2node<NodeDataT> : B
#pragma mtr node C2node<NodeDataT> : C
#pragma mtr node D2node<NodeDataT> : D
#pragma mtr node E2node<NodeDataT> : E
#pragma mtr node EndNode2 : End
#pragma mtr node sinkNodeAccept2 : SINK<PipeEltT>
//rep 3
#pragma mtr node A3node<NodeDataT> : A
#pragma mtr node B3node<NodeDataT> : B
#pragma mtr node C3node<NodeDataT> : C
#pragma mtr node D3node<NodeDataT> : D
#pragma mtr node E3node<NodeDataT> : E
#pragma mtr node EndNode3 : End
#pragma mtr node sinkNodeAccept3 : SINK<PipeEltT>
//rep 4
#pragma mtr node A4node<NodeDataT> : A
#pragma mtr node B4node<NodeDataT> : B
#pragma mtr node C4node<NodeDataT> : C
#pragma mtr node D4node<NodeDataT> : D
#pragma mtr node E4node<NodeDataT> : E
#pragma mtr node EndNode4 : End
#pragma mtr node sinkNodeAccept4 : SINK<PipeEltT>

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

// End2 -> SINK2
#pragma mtr edge EndNode2::accept -> sinkNodeAccept2

// Begin -> A3
#pragma mtr edge BeginNode::accept3 -> A3node

// A3-> B3
#pragma mtr edge A3node::accept -> B3node

// B3 -> C3
#pragma mtr edge B3node::accept -> C3node

// C3 -> D3
#pragma mtr edge C3node::accept -> D3node

// D3 -> E3
#pragma mtr edge D3node::accept -> E3node

// E3 -> End3
#pragma mtr edge E3node::accept -> EndNode3

// End3 -> SINK3
#pragma mtr edge EndNode3::accept -> sinkNodeAccept3

// Begin -> A4
#pragma mtr edge BeginNode::accept4 -> A4node

// A4-> B4
#pragma mtr edge A4node::accept -> B4node

// B4 -> C4
#pragma mtr edge B4node::accept -> C4node

// C4 -> D4
#pragma mtr edge C4node::accept -> D4node

// D4 -> E4
#pragma mtr edge D4node::accept -> E4node

// E4 -> End4
#pragma mtr edge E4node::accept -> EndNode4

// End4 -> SINK4
#pragma mtr edge EndNode4::accept -> sinkNodeAccept4
