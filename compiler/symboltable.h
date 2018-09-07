#ifndef __SYMBOLTABLE_H
#define __SYMBOLTABLE_H

#include <string>
#include <unordered_map>

class SymbolTable {
private:
  
  typedef std::unordered_map<std::string, int> SymMap;
  
public:
  
  enum { 
    NOT_FOUND = -1
  };
  
  size_t size() const { return map.size(); }

  bool insertUnique(std::string s, int i)
  {
    auto res = map.insert(std::make_pair(s, i));
    
    return res.second;
  }
  
  int find(const std::string &s)
  {
    auto it = map.find(s);
    
    if (it == map.end())
      return NOT_FOUND;
    else
      return it->second;
  }
  
private:
  
  SymMap map;
};

#endif
