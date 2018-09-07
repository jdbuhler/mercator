/**
  * Test input file for MERCATOR.
  * Tests a multistage pipeline with multiple sinks.
  *
  * Mapping: 2 elts per 1 thread.
  */

#include "datatypes.h"

/***  App name ***/
#pragma mtr application BlastApp_2to1map<BlastData>

/*** App-level data type. ***/
//#pragma mtr appdata UserDataExt
//#pragma mtr appdata BlastData

/*** Module (i.e., module type) specs. ***/

// Filter1 
//#pragma mtr module SeedMatch (int[512] -> outStream<point>:?1 | 1 : 1) 
#pragma mtr module SeedMatch (int[256] -> outStream<point>:?1 | 2 : 1) 
//#pragma mtr module SeedEnum (point[512] -> outStream<point>:?16 | 1 : 1) 
#pragma mtr module SeedEnum (point[256] -> outStream<point>:?16 | 2 : 1) 
#pragma mtr module SmallExt (point[256] -> outStream<point>:?1 | 2 : 1) 
#pragma mtr module UngapExt (point[256] -> outStream<point>:?1 | 2 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<int>

// SINK Module
#pragma mtr module SINK<point>


/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node seedMatchNode : SeedMatch
#pragma mtr node seedEnumNode : SeedEnum
#pragma mtr node smallExtNode : SmallExt
#pragma mtr node ungapExtNode : UngapExt
#pragma mtr node sinkNodeAccept : SINK<point>


/*** Edge specs. ***/

// SOURCE -> SeedMatch
#pragma mtr edge sourceNode::outStream -> seedMatchNode

// SeedMatch -> SeedEnum
#pragma mtr edge seedMatchNode::outStream -> seedEnumNode

// SeedEnum -> SmallExt
#pragma mtr edge seedEnumNode::outStream -> smallExtNode

// SmallExt -> UngapExt
#pragma mtr edge smallExtNode::outStream -> ungapExtNode

// UngapExt -> SINK
#pragma mtr edge ungapExtNode::outStream -> sinkNodeAccept

