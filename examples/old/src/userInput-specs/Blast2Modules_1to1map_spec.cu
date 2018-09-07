/**
  * Test input file for MERCATOR.
  * Tests a multistage pipeline with multiple sinks.
  *
  * Mapping: 1 elt to 1 thread.
  */

#include "datatypes.h"

/***  App name ***/
#pragma mtr application Blast2ModulesApp_1to1map<BlastData>

/*** App-level data type. ***/

/*** Module (i.e., module type) specs. ***/

// Filter1 
//#pragma mtr module SeedMatch (int[512] -> outStream<point>:?1 | 1 : 1) 
#pragma mtr module SeedMatch (int[128] -> outStream<point>:?1 | 1 : 1) 
#pragma mtr module SeedEnumAndExts (point[128] -> outStream<point>:?16 | 1 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<int>

// SINK Module
#pragma mtr module SINK<point>


/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node seedMatchNode : SeedMatch
#pragma mtr node seedEnumAndExtsNode : SeedEnumAndExts
#pragma mtr node sinkNodeAccept : SINK<point>


/*** Edge specs. ***/

// SOURCE -> SeedMatch
#pragma mtr edge sourceNode::outStream -> seedMatchNode

// SeedMatch -> SeedEnumAndExts
#pragma mtr edge seedMatchNode::outStream -> seedEnumAndExtsNode

// SeedEnumAndExts -> SINK
#pragma mtr edge seedEnumAndExtsNode::outStream -> sinkNodeAccept

