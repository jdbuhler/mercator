#ifndef __SIGNAL_CUH
#define __SIGNAL_CUH

class Signal {
private:
	int credit;

public:
	enum SignalTag {Tail, Enum, Agg};
	__device__ Signal() : credit(0) { }
	//__device__ Signal() { }
	__device__ ~Signal() { }

	__device__ void setCredit(int c) { credit = c; }
	__device__ int getCredit() { return credit; }

private:
	SignalTag tag;

public:
	__device__ void setTag(SignalTag t) { tag = t; }
	__device__ SignalTag getTag() { return tag; }
};

#endif
