/**
  * Input spec file for MERCATOR.
  * Specifies a pipeline consisting of five nodes, 
  *  plus one setup node and one cleanup (or base-case) node.
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
  *    Source -> Begin -> A -> B -> C -> D -> E -> End -> Sink
  *
  * Roles:
  *           Begin: setup node
  *           A-E: main processing nodes
  *           End: finish node
  *
  * Mapping: 4:1 from elements to threads; i.e., 4 elements
  *          are assigned to each thread
  */

// placeholder
struct PipeEltT {
};

/*** App name. ***/
#pragma mtr application DiffTypePipe_4to1map

/*** Module (i.e., module type) specs. ***/

// Setup
#pragma mtr module Begin (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module A (PipeEltT[512] -> accept<PipeEltT>:?1 | 4 : 1) 

#pragma mtr module B (PipeEltT[512] -> accept<PipeEltT>:?1 | 4 : 1) 

#pragma mtr module C (PipeEltT[512] -> accept<PipeEltT>:?1 | 4 : 1) 

#pragma mtr module D (PipeEltT[512] -> accept<PipeEltT>:?1 | 4 : 1) 

#pragma mtr module E (PipeEltT[512] -> accept<PipeEltT>:?1 | 4 : 1) 

#pragma mtr module End (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 


// SOURCE Module
#pragma mtr module SOURCE<PipeEltT>

// SINK Module
#pragma mtr module SINK<PipeEltT>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node BeginNode : Begin
#pragma mtr node Anode<NodeDataT> : A
#pragma mtr node Bnode<NodeDataT> : B
#pragma mtr node Cnode<NodeDataT> : C
#pragma mtr node Dnode<NodeDataT> : D
#pragma mtr node Enode<NodeDataT> : E
#pragma mtr node EndNode : End
#pragma mtr node sinkNodeAccept : SINK<PipeEltT>


/*** Edge specs. ***/

// SOURCE -> Begin
#pragma mtr edge sourceNode::outStream -> BeginNode

// Begin -> A
#pragma mtr edge BeginNode::accept -> Anode

// A -> B
#pragma mtr edge Anode::accept -> Bnode

// B -> C
#pragma mtr edge Bnode::accept -> Cnode

// C -> D
#pragma mtr edge Cnode::accept -> Dnode

// D -> E
#pragma mtr edge Dnode::accept -> Enode

// E -> End
#pragma mtr edge Enode::accept -> EndNode


// End -> SINK
#pragma mtr edge EndNode::accept -> sinkNodeAccept

