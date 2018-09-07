/**
  * Test input file for MERCATOR.
  * Tests frontend handling of back edges in different topologies.
  */


#if 1
  /**
  * App: TopoLoopTest1
  * Tests a multistage pipeline with multiple sinks,
  *  multiple module types, AND a filter with multiple
  *  elements per thread.
  * Topology: Source -> A -> B -> Sink1
  *           A -> Sink2
  *           B -> Sink3
  *           B -> A   (back edge)
  *
  * Topology should PASS frontend tests.
  */

/*** App name. ***/
#pragma mtr application TopoLoopTest1

/*** Module (i.e., module type) specs. ***/

// Filter1
// 4 inputs per thread
#pragma mtr module Filter1<MyModuleData> (int[32] -> accept<int>:1, reject<int>:1 | 1 : 1) 

// Filter2
#pragma mtr module Filter2<MyModuleData> (int[768] -> accept<int>:1, reject<int>:1, back<int>:1 | 1 : 1) 

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

// Filter2 -> Filter1   (back edge)
#pragma mtr edge filter2node::back -> filter1node

#endif


#if 0

  /**
  * App: Multiple loops.
  * Tests a multistage pipeline with multiple loops.
  * Topology: Source -> A1 -> B1 -> A2 -> B2 -> SINK
  *           B1 -> A1 (back edge)
  *           B2 -> A2 (back edge)
  *
  * Topology should PASS frontend tests.
  */

/*** App name. ***/
#pragma mtr application TopoLoopTest2

/*** Module (i.e., module type) specs. ***/

//  A: one output edge   
//  B: two output edges
#pragma mtr module A<int> (int[128] -> out<int>:1  | 1 : 1) 
#pragma mtr module B<int> (int[128] -> out1<int>:1, out2<int>:1  | 1 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<int>

// SINK Module
#pragma mtr module SINK<int>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node A1node : A
#pragma mtr node A2node : A
#pragma mtr node B1node : B
#pragma mtr node B2node : B
#pragma mtr node sinkNode : SINK<int>


/*** Edge specs. ***/

// SOURCE -> A1
#pragma mtr edge sourceNode::outStream -> A1node

// A1 -> B1
#pragma mtr edge A1node::out -> B1node

// B1 -> A2
#pragma mtr edge B1node::out1 -> A2node

// A2 -> B2
#pragma mtr edge A2node::out -> B2node

// B2 -> SINK
#pragma mtr edge B2node::out1 -> sinkNode

// B1 -> A1  (back edge)
#pragma mtr edge B1node::out2 -> A1node

// B2 -> A2  (back edge)
#pragma mtr edge B2node::out2 -> A2node

#endif

#if 0

  /**
  * App: Multiple loops that touch: tail of one is head of other.
  * Tests a multistage pipeline with multiple chained loops.
  * Topology: Source -> A1 -> B1 -> B2 -> SINK
  *           B1 -> A1 (back edge)
  *           B2 -> B1 (back edge)
  *
  * Topology should PASS frontend tests.
  */

/*** App name. ***/
#pragma mtr application TopoLoopTest3

/*** Module (i.e., module type) specs. ***/

//  A: one output edge   
//  B: two output edges
#pragma mtr module A<int> (int[128] -> out<int>:1  | 1 : 1) 
#pragma mtr module B<int> (int[128] -> out1<int>:1, out2<int>:1  | 1 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<int>

// SINK Module
#pragma mtr module SINK<int>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node A1node : A
#pragma mtr node B1node : B
#pragma mtr node B2node : B
#pragma mtr node sinkNode : SINK<int>


/*** Edge specs. ***/

// SOURCE -> A1
#pragma mtr edge sourceNode::outStream -> A1node

// A1 -> B1
#pragma mtr edge A1node::out -> B1node

// B1 -> B2
#pragma mtr edge B1node::out1 -> B2node

// B2 -> SINK
#pragma mtr edge B2node::out1 -> sinkNode

// B1 -> A1  (back edge)
#pragma mtr edge B1node::out2 -> A1node

// B2 -> B1  (back edge)
#pragma mtr edge B2node::out2 -> B1node

#endif


#if 0

  /**
  * App: Multiple NESTED loops.
  * Tests a multistage pipeline with nested loops.
  * Topology: Source -> A1 -> A2 -> B1 -> B2 -> SINK
  *           B1 -> A2 (back edge)
  *           B2 -> A1 (back edge)
  *
  * Topology should FAIL frontend tests.
  */

/*** App name. ***/
#pragma mtr application TopoLoopTest4

/*** Module (i.e., module type) specs. ***/

//  A: one output edge   
//  B: two output edges
#pragma mtr module A<int> (int[128] -> out<int>:1  | 1 : 1) 
#pragma mtr module B<int> (int[128] -> out1<int>:1, out2<int>:1  | 1 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<int>

// SINK Module
#pragma mtr module SINK<int>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node A1node : A
#pragma mtr node A2node : A
#pragma mtr node B1node : B
#pragma mtr node B2node : B
#pragma mtr node sinkNode : SINK<int>


/*** Edge specs. ***/

// SOURCE -> A1
#pragma mtr edge sourceNode::outStream -> A1node

// A1 -> A2
#pragma mtr edge A1node::out -> A2node

// A2 -> B1
#pragma mtr edge A2node::out -> B1node

// B1 -> B2
#pragma mtr edge B1node::out1 -> B2node

// B2 -> SINK
#pragma mtr edge B2node::out1 -> sinkNode

// B1 -> A2  (back edge)
#pragma mtr edge B1node::out2 -> A2node

// B2 -> A1  (back edge)
#pragma mtr edge B2node::out2 -> A1node

#endif


#if 0

  /**
  * App: Multiple OVERLAPPING loops.
  * Tests a multistage pipeline with overlapping loops.
  * Topology: Source -> A1 -> A2 -> B1 -> B2 -> SINK
  *           B1 -> A1 (back edge)
  *           B2 -> A2 (back edge)
  *
  * Topology should FAIL frontend tests.
  */

/*** App name. ***/
#pragma mtr application TopoLoopTest5

/*** Module (i.e., module type) specs. ***/

//  A: one output edge   
//  B: two output edges
#pragma mtr module A<int> (int[128] -> out<int>:1  | 1 : 1) 
#pragma mtr module B<int> (int[128] -> out1<int>:1, out2<int>:1  | 1 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<int>

// SINK Module
#pragma mtr module SINK<int>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node A1node : A
#pragma mtr node A2node : A
#pragma mtr node B1node : B
#pragma mtr node B2node : B
#pragma mtr node sinkNode : SINK<int>


/*** Edge specs. ***/

// SOURCE -> A1
#pragma mtr edge sourceNode::outStream -> A1node

// A1 -> A2
#pragma mtr edge A1node::out -> A2node

// A2 -> B1
#pragma mtr edge A2node::out -> B1node

// B1 -> B2
#pragma mtr edge B1node::out1 -> B2node

// B2 -> SINK
#pragma mtr edge B2node::out1 -> sinkNode

// B1 -> A2  (back edge)
#pragma mtr edge B1node::out2 -> A1node

// B2 -> A1  (back edge)
#pragma mtr edge B2node::out2 -> A2node

#endif


#if 0

  /**
  * App: Multiple OVERLAPPING loops, in tree structure.
  * Tests a tree with overlapping loops.
  * Topology: Source -> A1 -> A2 -> B1 -> B2 -> SINK1
  *                                 B1 -> B3 -> SINK2
  *           B2 -> A1 (back edge)
  *           B3 -> A2 (back edge)
  *
  * Topology should FAIL frontend tests.
  */

/*** App name. ***/
#pragma mtr application TopoLoopTest6

/*** Module (i.e., module type) specs. ***/

//  A: one output edge   
//  B: two output edges
#pragma mtr module A<int> (int[128] -> out<int>:1  | 1 : 1) 
#pragma mtr module B<int> (int[128] -> out1<int>:1, out2<int>:1  | 1 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<int>

// SINK Module
#pragma mtr module SINK<int>

/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node A1node : A
#pragma mtr node A2node : A
#pragma mtr node B1node : B
#pragma mtr node B2node : B
#pragma mtr node B3node : B
#pragma mtr node sinkNode1 : SINK<int>
#pragma mtr node sinkNode2 : SINK<int>


/*** Edge specs. ***/

// SOURCE -> A1
#pragma mtr edge sourceNode::outStream -> A1node

// A1 -> A2
#pragma mtr edge A1node::out -> A2node

// A2 -> B1
#pragma mtr edge A2node::out -> B1node

// B1 -> B2
#pragma mtr edge B1node::out1 -> B2node

// B1 -> B3
#pragma mtr edge B1node::out2 -> B3node

// B2 -> SINK
#pragma mtr edge B2node::out1 -> sinkNode1

// B3 -> SINK
#pragma mtr edge B3node::out1 -> sinkNode2

// B2 -> A1  (back edge)
#pragma mtr edge B2node::out2 -> A1node

// B3 -> A2  (back edge)
#pragma mtr edge B3node::out2 -> A2node

#endif

#if 0
  /**
  * App: TopoLoopTest7
  * Tests a multistage pipeline with multiple sinks,
  *  multiple module types, AND a filter with multiple
  *  elements per thread, with production rate greater than 1 on the cycle.
  * Topology: Source -> A -> B -> Sink1
  *           A -> Sink2
  *           B -> Sink3
  *           B -> A   (back edge)
  *
  * Topology should FAIL frontend tests b/c of production rate > 1 on cycle.
  */

/*** App name. ***/
#pragma mtr application TopoLoopTest7

/*** Module (i.e., module type) specs. ***/

// Filter1
// 4 inputs per thread
#pragma mtr module Filter1<MyModuleData> (int[32] -> accept<int>:2, reject<int>:1 | 1 : 1) 

// Filter2
#pragma mtr module Filter2<MyModuleData> (int[768] -> accept<int>:1, reject<int>:1, back<int>:1 | 1 : 1) 

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

// Filter2 -> Filter1   (back edge)
#pragma mtr edge filter2node::back -> filter1node

#endif
