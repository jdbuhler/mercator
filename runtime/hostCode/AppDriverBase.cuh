#ifndef __APPDRIVERBASE_CUH
#define __APPDRIVERBASE_CUH

//
// @file AppDriverBase.cuh
// A virtual base class for the app driver, which insulates
// the host app's .cuh file, and hence the user's host-side codebase,
// from having to include any device app code.
//

namespace Mercator  {

  template <typename HostParamsT>
  class AppDriverBase {
  public:
    virtual ~AppDriverBase() {}
    
    virtual void runAsync(const HostParamsT *params) = 0;
    
    virtual void join() = 0;
    
    // @brief synchronous run in terms of runAsync + join
    void run(const HostParamsT *params)
    {
      runAsync(params);
      join();
    }
  };
  
}
    
#endif
