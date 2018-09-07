#ifndef __TOPOVERIFY_H
#define __TOPOVERIFY_H

#include "app.h"

class TopologyVerifier {
public:
  
  static
  void verifyTopology(App *app);
  
private:
  
  static
  Node *dfsVisit(Node *node,
		 Edge *parentEdge,
		 long multiplier);
  
};

#endif
