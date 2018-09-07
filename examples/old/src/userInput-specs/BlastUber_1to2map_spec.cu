/**
  *
  * Mapping: 1 elt per 2 threads.
  */

#include "datatypes.h"

/***  App name ***/
#pragma mtr application BlastUberApp_1to2map<BlastData>

/*** App-level data type. ***/
//#pragma mtr appdata UserDataExt
//#pragma mtr appdata BlastData

/*** Module (i.e., module type) specs. ***/

// Filter1 
//#pragma mtr module SeedMatch (int[512] -> outStream<point>:?1 | 1 : 1) 
//#pragma mtr module SeedMatch (int[128] -> outStream<point>:?1 | 2 : 1) 
#pragma mtr module FullBlast (int[128] -> outStream<point>:?1 | 1 : 2) 
//#pragma mtr module SeedEnum (point[512] -> outStream<point>:?16 | 1 : 1) 
//#pragma mtr module SeedEnum (point[128] -> outStream<point>:?16 | 2 : 1) 
//#pragma mtr module SmallExt (point[2048] -> outStream<point>:?1 | 2 : 1) 
//#pragma mtr module UngapExt (point[2048] -> outStream<point>:?1 | 2 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<int>

// SINK Module
#pragma mtr module SINK<point>


/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node fullBlastNode : FullBlast
//#pragma mtr node seedMatchNode : SeedMatch
//#pragma mtr node seedEnumNode : SeedEnum
//#pragma mtr node smallExtNode : SmallExt
//#pragma mtr node ungapExtNode : UngapExt
#pragma mtr node sinkNodeAccept : SINK<point>


/*** Edge specs. ***/

// SOURCE -> FullBlast
#pragma mtr edge sourceNode::outStream -> fullBlastNode

// FullBlast -> SINK
#pragma mtr edge fullBlastNode::outStream -> sinkNodeAccept

