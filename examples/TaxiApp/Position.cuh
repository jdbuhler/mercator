#ifndef POSITION_CUH
#define POSITION_CUH

class Position {
	public:
		//unsigned char tag[8];
		unsigned int tag;
		double longitude;
		double latitude;

		__device__ __host__
		Position() {
			longitude = 0.0;
			latitude = 0.0;
			//tag[0] = '\0';
			tag = 0;
		}

		__device__ __host__
		//Position(unsigned char* t, double lon, double lat) {
		Position(unsigned int t, double lon, double lat) {
			longitude = lon;
			latitude = lat;
			//for(unsigned int j = 0; j < 8; ++j) {
			//	tag[j] = t[j];
			//}
			tag = t;
		}
};

#endif
