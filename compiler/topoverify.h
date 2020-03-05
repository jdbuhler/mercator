#ifndef __TOPOVERIFY_H
#define __TOPOVERIFY_H

//
// TOPOVERIFY.H
// Verify that a MERCATOR application has legal topology, and
// compute some properties of that topology.
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "app.h"

class TopologyVerifier {
public:
  
  static
  void verifyTopology(App *app);
  
private:

  static int currentId;
  
  static
  Node *dfsVisit(Node *node,
		 Edge *parentEdge,
		 long multiplier,
		 int enumId,
		 App *app);
  
};

#endif
