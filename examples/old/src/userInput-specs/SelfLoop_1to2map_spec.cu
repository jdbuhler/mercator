/**
  * Input spec file for MERCATOR.
  * Specifies a pipeline consisting of three nodes: 
  *  a trivial setup node, a node with a self-loop,
  *  and a finish node.
  * This topology, when combined with input-specific
  *  routing data, is sufficient to implement many
  *  classes of applications, such as recursion trees
  *  of arbitrary depth and other tree-traversal apps,
  *  such as random forest evaluation.
  *  
  * Topology: Source -> Begin -> A -> End -> Sink
  *           A -> A   (self-loop)
  *
  * Roles:
  *           Begin: setup node
  *           A: main processing node
  *           End: finish node
  *
  * Mapping: 1:2 from elements to threads; i.e., 2 threads
  *          are assigned to each element
  */

// placeholder
struct PipeEltT {
};

/*** App name. ***/
#pragma mtr application SelfLoop_1to2map

/*** Module (i.e., module type) specs. ***/

// Filter1

// Filter2
//#pragma mtr module A<MyModuleData> (PipeEltT[32] ->
//accept<PipeEltT>:1, reject<PipeEltT>:1 | 1 : 1) 
#pragma mtr module Begin (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

#pragma mtr module A (PipeEltT[128] -> accept<PipeEltT>:?1, self<PipeEltT>:?1 | 1 : 2) 

#pragma mtr module End (PipeEltT[128] -> accept<PipeEltT>:?1 | 1 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<PipeEltT>

// SINK Module
#pragma mtr module SINK<PipeEltT>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
//#pragma mtr node Anode<MyNodeData> : A
#pragma mtr node BeginNode : Begin
#pragma mtr node Anode<NodeData5FiltT> : A
#pragma mtr node EndNode : End
#pragma mtr node sinkNodeAccept : SINK<PipeEltT>


/*** Edge specs. ***/

// SOURCE -> Begin
#pragma mtr edge sourceNode::outStream -> BeginNode

// Begin -> A
#pragma mtr edge BeginNode::accept -> Anode

// A -> A
#pragma mtr edge Anode::self -> Anode

// A -> End
#pragma mtr edge Anode::accept -> EndNode

// End -> SINK
#pragma mtr edge EndNode::accept -> sinkNodeAccept

