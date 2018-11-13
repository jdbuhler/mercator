#ifndef __SIGNAL_CUH
#define __SIGNAL_CUH

class Signal {
private:
	bool start;
	bool end;
	bool tail;
	int credit;

public:
	__device__ Signal() : start(false), end(false), tail(false), credit(0) { }
	//__device__ Signal() { }
	__device__ ~Signal() { }

	__device__ void setStart(bool s) { start = s; }
	__device__ void setEnd(bool e) { end = e; }
	__device__ void setTail(bool t) { tail = t; }
	__device__ void setCredit(int c) { credit = c; }

	__device__ bool getStart() { return start; }
	__device__ bool getEnd() { return end; }
	__device__ bool getTail() { return tail; }
	__device__ int getCredit() { return credit; }
};

#endif
