//
// TYPECHECK.CC
// Interface to LLVM type checker
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#include "typecheck.h"

#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "build_config.h"

using namespace std;
using namespace clang;

typedef unordered_map<string, QualType> TypeMap;

typedef unordered_map<string, string> StrMap;

/*
 * @class MercatorVisitor  *
 * @brief Traverses an ASTContext and constructs a mapping of string to
 *    QualType used in type comparison
 */
class MercatorVisitor : public RecursiveASTVisitor<MercatorVisitor> {
public:
  
  /*
   * @brief Constructor, assigns the ASTContext
   */
  MercatorVisitor(ASTContext* a,
		  const StrMap &ioriginalTypeStrings)
    : astContext(a),
      originalTypeStrings(ioriginalTypeStrings)
  {}
  
  /*
   * @brief Visits a variable declaration found in the ASTContext. 
   *   Looks at VarDecls in the top file ONLY.
   * @param var Pointer to the currently found VarDecl
   * @return bool Parse of VarDecl was successful or not
   */
  bool VisitVarDecl(VarDecl *var) 
  {
    // Check for VarDecls in the top (main) file ONLY
    if (astContext->getSourceManager().isInMainFile(var->getLocStart())) 
      {
	// For each variable that we created for type-checking purposes,
	// map its typename *as it appears in the specfile* to
	// its QualType as determined by Clang.
	const string varName = var->getNameAsString();
	const string origTypeString = originalTypeStrings.find(varName)->second;
	
	typeIdx.insert(make_pair(origTypeString, var->getType()));
      }
    
    return true;
  }
  
  // access the type map inferred from the AST
  const TypeMap &get_typeIdx() const { return typeIdx; }
  
private:
  
  // Pointer to current ASTContext
  ASTContext *astContext;
  
  // original type strings as they appear in the spec file
  const StrMap &originalTypeStrings;
  
  //Map of all the QualTypes found in astContext and their associated names
  TypeMap typeIdx;
  
}; // end MercatorVisitor class


/*
 * @class ASTContainerImpl
 * @brief Constructs the ASTContext using the MercatorVisitor and allows the user to compare types.
 */
class ASTContainerImpl : public ASTContainer {
public:
  
  ASTContainerImpl() 
    : vst(nullptr)
  {}
  
  ~ASTContainerImpl()
  {
    if (vst)
      delete vst;
  }
  
  /*
   * @brief Finds all the types in the current ASTContext (stored in the 
   *  MercatorVisitor vst)
   *
   * @param typeStrings The names of the types that were found when parsing
   */
  void findTypes(const vector<string> &typeStrings,
		 const vector<string> &references,
		 const vector<string> &includePaths)
  {
    string code = "";
    
    // add all references to the current code
    for (const string &ref : references)
      code += "#include \"" + ref + "\"\n";
    
    // add dummy variables for ASTContext to be able to grab QualTypes easily
    int dummyNumber = 0;
    code = code + "\nvoid __typecheck_dummyDecls() {\n";
    for (const string &typeString : typeStrings)
      {
	if (typeString != "NULL") 
	  {
	    string varName = "__typecheck_dummy" + to_string(dummyNumber);
	    originalTypeStrings.insert(make_pair(varName, typeString)); 
	    
	    code += typeString + " " + varName + ";\n";
	    ++dummyNumber;
	  }
      }

    // stimcheck: Make sure that an unsigned int exists for the compiler to see.
    // This is for when we create enumeration and aggregation nodes.
    string varName = "__typecheck_dummy" + to_string(dummyNumber);
    originalTypeStrings.insert(make_pair(varName, "unsigned int")); 
	    
    code += "unsigned int " + varName + ";\n";
    ++dummyNumber;

    code = code + "}";
    
    //Set up args for building the ASTContext 
    vector<string> args;
    
    args.push_back("--std=c++11");
    
    // enable for path debugging
    // args.push_back("-v");
    
    // we must include the path to Clang's standard system libraries
    args.push_back("-I" LLVM_PATH "/lib/clang/" LLVM_VERSION "/include");

    // include the paths supplied by local CUDA install, which could
    // be a list separated by ";" characters
    string cudaIncs = CUDA_INCLUDE_PATH;
    
    size_t start = 0;
    while (true)
      {
	size_t pos = cudaIncs.find_first_of(";", start);
	size_t len;
	if (pos == string::npos)
	  len = string::npos;
	else
	  len = pos - start;
	
	args.push_back("-I" + cudaIncs.substr(start, len));
	
	if (pos == string::npos)
	  break;
	else
	  start = pos + 1;
      }
    
    for (const string &incPath : includePaths)
      args.push_back("-I" + incPath);
    
    // Build the ASTContext with the code generated and with the args.
    // NB: all Clang compilation errors occur from within this call.
    // But there is no obvious way for us to hook into those errors,
    // so it's not clear how we know if the compilation failed.  Type
    // errors do not cause Clang to return a null AST.
    
    ast = clang::tooling::buildASTFromCodeWithArgs(code, args);
    
    if (ast == nullptr)
      {
	cerr << "ERROR: LLVM/Clang failed to typecheck specfile\n";
	abort();
      }
    
    //Set the ASTContext of the ASTContainerImpl
    ASTContext *pctx = &(ast->getASTContext());
    vst = new MercatorVisitor(pctx, originalTypeStrings);
    
    //Get a pointer to the TranslationUnitDecl for traversing ASTContext
    TranslationUnitDecl* decl = pctx->getTranslationUnitDecl();
    
    //Visit all the nodes in the current ASTContext with the MercatorVisitor.
    vst->TraverseDecl(decl);
    
#if 0
    //Print all the types found in the ASTContext
    printTypeComparisions();    
#endif
  }
  
  /*
   * @brief Checks whether a type exists in the current ASTContext
   *
   * @param typeName The name of the type being queried
   * 
   * @return bool True if the type exists, False otherwise
   */
  bool queryType(const string &typeName) const
  {
    const TypeMap &typeIdx = vst->get_typeIdx();
    
    return (typeIdx.find(typeName) != typeIdx.end());
  }
  
  
  /*
   * @brief map a typename to an integer that is unique for
   * the corresponding canonical type.  The typename must be
   * valid.
   *
   * @ param typeName name of type to look up
   * @return integer unique to this typeName's canonical type
   */
  unsigned long typeId(const string &typeName) const
  {
    const TypeMap &typeIdx = vst->get_typeIdx();
    
    assert(queryType(typeName));
    
    return (unsigned long) 
      typeIdx.at(typeName).getCanonicalType().getAsOpaquePtr();
  }
  
  /*
   * @brief Compares two types of the names provided
   *
   * @param firstName The first type's name
   * @param secondName The second type's name
   *
   * @return bool If the canonical types of the two names are the same
   * returns True, otherwise False
   */
  bool compareTypes(const string &firstName, 
		    const string &secondName) const
  {
    const TypeMap &typeIdx = vst->get_typeIdx();
    
    // TypeMap::at will crash unless the value exists!
    assert(typeIdx.find(firstName)     != typeIdx.end()
	   && typeIdx.find(secondName) != typeIdx.end());
    
    return (typeIdx.at(firstName).getCanonicalType() == 
	    typeIdx.at(secondName).getCanonicalType());
  }

private:
  
  std::unique_ptr<ASTUnit> ast;	// Pointer to AST that holds type info
  StrMap originalTypeStrings;   // type strings as they appear in the spec file
  MercatorVisitor *vst;	        // Visitor object holding type map
  
  /*
   * @brief Prints all the string names and QualTypes (as strings) and
   * their comparisons in the typeIdxMap to cout.
   */
  void printTypeComparisions() const
  {
    cout << "TYPE COMPARISONS" << endl;
    for (auto& it : vst->get_typeIdx())
      {
	cout << endl << it.first << '\t' << it.second.getAsString() << endl;
	cout << "---------------------------" << endl;
	
	for (auto& itt : vst->get_typeIdx())
	  {
	    cout << itt.first << '\t' << itt.second.getAsString() << '\t' 
		 << compareTypes(itt.first, it.first) << endl;
	  }
      }
  }

};

ASTContainer *ASTContainer::create() 
{ return new ASTContainerImpl(); }
