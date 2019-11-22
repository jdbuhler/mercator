#ifndef TAGGED_POSITION_CUH
#define TAGGED_POSITION_CUH

class TaggedPosition {
	public:
		unsigned int tag;
		const char* pos;

		__device__ __host__
		TaggedPosition() {
			tag = 0;
			pos = 0;
		}

		__device__ __host__
		TaggedPosition(unsigned int t, const char* p) {
			tag = t;
			pos = p;
		}
};

#endif
