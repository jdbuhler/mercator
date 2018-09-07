/**
  *
  * Mapping: 2 elts per 1 thread.
  */

#include "datatypes.h"

/***  App name ***/
#pragma mtr application BlastUberApp_2to1map<BlastData>

/*** App-level data type. ***/

/*** Module (i.e., module type) specs. ***/

/////// Break main functionality into two parts because of the replication in the SeedEnum stage
//#pragma mtr module SeedMatchAndEnum (int[128] -> outStream<point>:?16 | 1 : 1) 
//#pragma mtr module SmallAndUngapExt (point[128] -> outStream<point>:?1 | 1 : 1) 
#pragma mtr module FullBlast (int[128] -> outStream<point>:?16 | 2 : 1) 

// SOURCE Module
#pragma mtr module SOURCE<int>

// SINK Module
#pragma mtr module SINK<point>


/*** Node (i.e., module instance) specs. ***/
#pragma mtr node sourceNode : SOURCE
#pragma mtr node fullBlastNode : FullBlast
#pragma mtr node sinkNodeAccept : SINK<point>


/*** Edge specs. ***/

// SOURCE -> FullBlast
#pragma mtr edge sourceNode::outStream -> fullBlastNode


// FullBlast -> SINK
#pragma mtr edge fullBlastNode::outStream -> sinkNodeAccept

