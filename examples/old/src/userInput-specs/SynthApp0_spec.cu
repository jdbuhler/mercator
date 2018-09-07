/**
  * Test input file for MERCATOR.
  * Tests a multistage pipeline with multiple sinks,
  *  AND multiple module types.
  * NB: this app is an extension of App0 with multiple 
  *      module types.
  * Thread/item mapping is 1/1 for all modules.
  * Topology: Source -> A -> B -> Sink1
  *           A -> Sink2
  *           B -> Sink3
  */

/*** App name. ***/
#pragma mtr application SynthApp0

/*** Module (i.e., module type) specs. ***/

// Filter1
#pragma mtr module Filter1<MyModuleData> (int[32] -> accept<int>:4, reject<int>:4 | 1 : 1) 

// Filter2
#pragma mtr module Filter2<MyModuleData> (int[32] -> accept<int>:4, reject<int>:4 | 1 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<int>

// SINK Module
#pragma mtr module SINK<int>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node filter1node<MyNodeData> : Filter1
#pragma mtr node filter2node<MyNodeData> : Filter2
#pragma mtr node sinkNodeReject1 : SINK<int>
#pragma mtr node sinkNodeReject2 : SINK<int>
#pragma mtr node sinkNodeAccept : SINK<int>


/*** Edge specs. ***/

// SOURCE -> Filter1
#pragma mtr edge sourceNode::outStream -> filter1node

// Filter1 -> Filter2
#pragma mtr edge filter1node::accept -> filter2node

// Filter1 -> Rejects
#pragma mtr edge filter1node::reject -> sinkNodeReject1

// Filter2 -> SINK
#pragma mtr edge filter2node::accept -> sinkNodeAccept

// Filter2 -> Rejects
#pragma mtr edge filter2node::reject -> sinkNodeReject2
