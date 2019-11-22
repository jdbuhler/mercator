#ifndef LINE_CUH
#define LINE_CUH

class Line {
	public:
		const char* startPointer;
		unsigned int length;
		unsigned int tag;

		__device__ __host__
		Line() {
			tag = 0;
			startPointer = 0;
			length = 0;
		}

		__device__ __host__
		Line(unsigned int t, const char* sp, unsigned int l) {
			tag = t;
			startPointer = sp;
			length = l;
		}
};

#endif
