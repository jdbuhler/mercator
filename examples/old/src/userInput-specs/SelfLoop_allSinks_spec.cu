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
  * Topology: Source -> A -> B -> C -> Sink
  *           B -> B   (self-loop)
  *           A, B, C -> rejectSinkA,B,C
  * Roles:
  *           A: setup node
  *           B: main processing node
  *           C: finish node
  */

// placeholder
struct SelfLoopEltT {
};

/*** App name. ***/
#pragma mtr application SelfLoop

/*** Module (i.e., module type) specs. ***/

// Filter1

// Filter2
//#pragma mtr module A<MyModuleData> (SelfLoopEltT[32] ->
//accept<SelfLoopEltT>:1, reject<SelfLoopEltT>:1 | 1 : 1) 
#pragma mtr module A (SelfLoopEltT[32] -> accept<SelfLoopEltT>:1, reject<SelfLoopEltT>:1 | 1 : 1) 

#pragma mtr module B (SelfLoopEltT[32] -> accept<SelfLoopEltT>:1, reject<SelfLoopEltT>:1, self<SelfLoopEltT>:1 | 1 : 1) 

#pragma mtr module C (SelfLoopEltT[32] -> accept<SelfLoopEltT>:1, reject<SelfLoopEltT>:1 | 1 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<SelfLoopEltT>

// SINK Module
#pragma mtr module SINK<SelfLoopEltT>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
//#pragma mtr node Anode<MyNodeData> : A
#pragma mtr node Anode : A
#pragma mtr node Bnode : B
#pragma mtr node Cnode : C
#pragma mtr node sinkNodeRejectA : SINK<SelfLoopEltT>
#pragma mtr node sinkNodeRejectB : SINK<SelfLoopEltT>
#pragma mtr node sinkNodeRejectC : SINK<SelfLoopEltT>
#pragma mtr node sinkNodeAccept : SINK<SelfLoopEltT>


/*** Edge specs. ***/

// SOURCE -> A
#pragma mtr edge sourceNode::outStream -> Anode

// A -> B
#pragma mtr edge Anode::accept -> Bnode

// A -> Rejects
#pragma mtr edge Anode::reject -> sinkNodeRejectA

// B -> C
#pragma mtr edge Bnode::accept -> Cnode

// B -> Rejects
#pragma mtr edge Bnode::reject -> sinkNodeRejectB

// B -> B
#pragma mtr edge Bnode::self -> Bnode

// C -> SINK
#pragma mtr edge Cnode::accept -> sinkNodeAccept

// C -> Rejects
#pragma mtr edge Cnode::reject -> sinkNodeRejectC

