#ifndef __BLASTUBER_DRIVER_CU
#define __BLASTUBER_DRIVER_CU

/**
 * @brief Driver (test harnesses) for Mercator app
 *          BlastApp.
 */

#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <limits.h>
#include <vector>

#include "./tests/datatypes.h"
#include "./tests/blastData.cuh"

#include "driver_config.cuh"

#if MAPPING_1TO1
  #include "../codegenInput/BlastUberApp_1to1map.cuh"
  #define APP_TYPE BlastUberApp_1to1map
#elif MAPPING_1TO2
  #include "../codegenInput/BlastUberApp_1to2map.cuh"
  #define APP_TYPE BlastUberApp_1to2map
#elif MAPPING_1TO4
  #include "../codegenInput/BlastUberApp_1to4map.cuh"
  #define APP_TYPE BlastUberApp_1to4map
#elif MAPPING_2TO1
  #include "../codegenInput/BlastUberApp_2to1map.cuh"
  #define APP_TYPE BlastUberApp_2to1map
#elif MAPPING_4TO1
  #include "../codegenInput/BlastUberApp_4to1map.cuh"
  #define APP_TYPE BlastUberApp_4to1map
#else
  #error "INVALID MAPPING SELECTION"
#endif

// BLAST-specific param
#define MAX_DIFF 128

#define PRINT_OUTPUT_BUFFERS_BLAST 0

#if 0
struct node {
	int x;
	node* next;
};
#endif

void run_blastUberApp()
{
	const int HASH_SIZE = 65536;
  // set up input buffer
  //const int BUFFER_CAPACITY = 1024;

	//Read database and query files
        // NB: Query file should be in text format with chars 'ACGT' only.
        //     DB file should be in packed format, with 2 bits per base (4 chars per byte), 
        //       using the following encoding: 00 = 'A', 01 = 'C',
        //       10 = 'G', 11 = 'T'

//	std::string queryFilename = "./bin/BlastData/query.txt"; // NB: this one works!
//	std::string dbFilename = "./bin/BlastData/d2.txt";

//	std::string queryFilename = "./bin/BlastData/query-replicated.txt"; // NB: this one works!
//	std::string dbFilename = "./bin/BlastData/d2-replicated.txt";

//	std::string queryFilename = "./bin/BlastData/salmonella.txt";
//	std::string queryFilename = "./bin/BlastData/salmonella-5k.txt";
//	std::string queryFilename = "./bin/BlastData/salmonella-2k.txt";
//	std::string queryFilename = "./bin/BlastData/salmonella-4k.txt";
//	std::string queryFilename = "./bin/BlastData/salmonella-6k.txt";
//	std::string queryFilename = "./bin/BlastData/salmonella-8k.txt";
//	std::string queryFilename = "./bin/BlastData/salmonella-10k.txt";
//	std::string queryFilename = "./bin/BlastData/salmonella-20k.txt";
//	std::string queryFilename = "./bin/BlastData/salmonella-30k.txt";
//	std::string queryFilename = "./bin/BlastData/salmonella-40k.txt";
//	std::string queryFilename = "./bin/BlastData/salmonella-50k.txt";
        
//	std::string dbFilename = "./bin/BlastData/ecoli-k12-231k-binary.txt";
//	std::string dbFilename = "./bin/BlastData/ecoli-k12-binary.txt";

	std::string db = "";
	std::string query = "";
	std::string tmp = "";

	std::ifstream queryFile;
        std::string queryFileName = QUERYFILENAME;
	queryFile.open(queryFileName.c_str(), std::ifstream::binary);

	std::ifstream dbFile;
        std::string dbFileName = DBFILENAME;
	dbFile.open(dbFileName.c_str(), std::ifstream::binary);

	queryFile.seekg(0, queryFile.end);
	int len = queryFile.tellg();
	char* cquery = new char[len];
	queryFile.seekg(0, queryFile.beg);
	queryFile.read(cquery, len);
	query = std::string(cquery, len);
	delete[] cquery;
	queryFile.close();

	dbFile.seekg(0, dbFile.end);
	len = dbFile.tellg();
	char* cdb = new char[len];
	dbFile.seekg(0, dbFile.beg);
	dbFile.read(cdb, len);
	db = std::string(cdb, len);
	delete[] cdb;
	dbFile.close();

	const int BUFFER_CAPACITY = len;

////////////////////////////////////
#if 1
  // print experiment params if desired
  // NB: all possible topos included for sanity check
  // convert topology indicators to string
#if RUN_BLASTAPP
  const char topoString[] = "BLAST";
#elif RUN_BLASTUBERAPP
  const char topoString[] = "BLASTUBER";
#elif RUN_BLAST2MODULESAPP
  const char topoString[] = "BLAST2Modules";
#else
  const char topoString[] = "NONSTANDARD";
#endif

      // convert mapping indicators to string
#if MAPPING_1TO1
      const char mapString[] = "1-to-1";
#elif MAPPING_1TO2
      const char mapString[] = "1-to-2";
#elif MAPPING_1TO4
      const char mapString[] = "1-to-4";
#elif MAPPING_2TO1
      const char mapString[] = "2-to-1";
#elif MAPPING_4TO1
      const char mapString[] = "4-to-1";
#else
      const char mapString[] = "NONSTANDARD";
#endif

      // print app metadata
      printf("APP PARAMS: TOPOLOGY: %s ELTS-TO-THREADS MAPPING: %s FILTER_RATE: %.2f INPUTS: %d\n", 
          topoString, mapString, FILTER_RATE, len); 

      //debug
//      printf("Size of input item: %d\n", sizeof(PipeEltT));
#endif
////////////////////////////////////

	//Construct CPU Query Hash Table
	node** queryHashes = new node*[HASH_SIZE];
	node* n;

	//node* a;
	for(int i = 0; i < HASH_SIZE; ++i) {
		//a = new node;
		//a->x = -1;
		//a->next = NULL;
		queryHashes[i] = NULL;
	}

	for(int i = 0; i < query.size() - 9; ++i) {
		//int word = ((int)((unsigned char)query.at(i))) * 256 + (unsigned char)query.at(i + 1);
		int word = 0;
		for(int j = 0; j < 8; ++j) {
			word = word << 2;
			char curr = query.at(i + j);
			switch(curr) {
				case 'G':
					//G
					word += 3;
					break;
				case 'T':
					//T
					word += 2;
					break;
				case 'C':
					//C
					word += 1;
					break;
				case 'A':
					//A
					word += 0;
					break;
				default:
					break;
			};
		}
		node* n2 = new node;
		n2->x = i;
		n2->next = queryHashes[word];
		queryHashes[word] = n2;
	}

	  int* inBufferData;
	  cudaMallocManaged(&inBufferData, BUFFER_CAPACITY * sizeof(int));
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );

	  Mercator::InputBuffer<int>* inBuffer = new Mercator::InputBuffer<int>(inBufferData, BUFFER_CAPACITY);

//	  std::string sourceNodeName = "sourceNode";

	  // set up output buffers
//	  std::vector<Mercator::IO::BufferBase*> outBufferVec;
//	  std::vector<std::string>               sinkNodeNameVec;

	  Mercator::OutputBuffer<point>* outBuff = new Mercator::OutputBuffer<point>(BUFFER_CAPACITY);

//	  outBufferVec.push_back(outBuff);
//	  std::string sinkNodeName = "sinkNodeAccept";
//	  sinkNodeNameVec.push_back(sinkNodeName);

	  // fill input buffer
	  for(int i=0; i < BUFFER_CAPACITY; ++i)
	    inBuffer->add(i);

	// print contents of input buffer
	//for(int i=0; i < inBuffer->get_numItems(); ++i)
	 // std::cout << "Input[" << i << "]: " << inBuffer->peek(i) << "\n";

	  // create app object
//	  Mercator::App* blastApp = new Mercator::App();
          APP_TYPE* blastApp = new APP_TYPE();

		//Construct GPU Query Hash Table
		int* qHits = new int[HASH_SIZE];
		int* qHash = new int[HASH_SIZE];
		for(int i = 0; i < 2; ++i) {
			qHits[i] = -1;
		}

		std::vector<int> vec = std::vector<int>();
		int pos = 2;
		for(int i = 0; i < HASH_SIZE; ++i){
			n = queryHashes[i];
			if(!n) {
				//no hits
				qHash[i] = -1;
			}
			else if(!n->next) {
				//single hit
				qHash[i] = n->x;
				//std::cout << qHash[i] << " ";
			}
			else {				
				//multiple hits
				//n = n->next;
				while(n) {
					vec.push_back(n->x);
					n = n->next;
				}
				//do array stuffs
				int* newarr = new int[pos + vec.size() + 1];
				memcpy(newarr, qHits, sizeof(int) * pos);
				for(int j = 0; j < vec.size(); ++j) {
					newarr[pos + j] = vec.at(j);
					//if(vec.at(j) == 48421) {
					//	std::cout << "H " << i << std::endl;
					//	std::string rsp3;
					//	std::cin >> rsp3;
					//}
				}
				newarr[pos + vec.size()] = -1;
				delete[] qHits;
				qHits = newarr;
				qHash[i] = pos * -1;
				pos += vec.size() + 1;
				//std::cout << vec.size() << ": ";
				//for(int j = 0; j < vec.size(); ++j)
				//	std::cout << vec.at(j) << " ";
				//std::cout << std::endl;
				vec = std::vector<int>();
				//std::string rsp;
				//std::cin >> rsp;
			}
			//std::cout << i << ": " << qHash[i] << std::endl;
			//if(i % 1000 == 0) {
			//	std::string rsp;
			//	std::cin >> rsp;
			//}
		}
		//std::cout << qHash[0x2bcd] << std::endl;
		/*
		int p = 0;
		int pMax = 0;
		for(int i = 0; i < pos; ++i) {
			if(qHits[i] == -1) {
				std::cout << qHits[i] << std::endl;
				pMax = max(p, pMax);
				p = 0;
			}
			else {
				std::cout << qHits[i] << " ";
				++p;
			}
		}
		std::cout << "pMAX = " << pMax << std::endl;
		std::string rsp2;
		std::cin >> rsp2;
		*/
		
		//Allocate c strings for query and databse (to go to GPU)
		int qSize = query.size();
		int dSize = db.size();
		Base* q = (Base*)malloc((qSize + STRING_BUFF * 8) * sizeof(Base));
		Base* d = (Base*)malloc((dSize + STRING_BUFF * 2) * sizeof(Base));

		//Buffer beginning of c strings
		for(int i = 0; i < STRING_BUFF; ++i) {
			for(int j = 0; j < 4; ++j) {
				q[i] = 'A';
			}
			d[i] = 0xFF;
		}

		//Fill query c string
		for(int i = STRING_BUFF * 4; i < qSize + STRING_BUFF * 4; ++i) {
			q[i] = query.at(i - STRING_BUFF * 4);
		}

		//Fill database c string
		for(int i = STRING_BUFF; i < dSize + STRING_BUFF; ++i) {
			d[i] = db.at(i - STRING_BUFF);
		}

		//Buffer end of c strings
		for(int i = 0; i < STRING_BUFF; ++i) {
			for(int j = 0; j < 4; ++j) {
				q[i + qSize + STRING_BUFF * 4] = 'A';
			}
			d[i + dSize + STRING_BUFF] = 0xFF;
		}

		//Initialize global app data on GPU
		BlastData* blastAppData = new BlastData(qHits, qHash, pos, HASH_SIZE, q, d, qSize + STRING_BUFF * 8, dSize + STRING_BUFF * 2);
		blastApp->set_userData(blastAppData);


  // associate buffers with nodes
  blastApp->sourceNode->set_inBuffer(inBuffer);
  blastApp->sinkNodeAccept->set_outBuffer(outBuff);

// print contents of output buffer
std::cout << "Calling run fcn for blast app... \n" ;

  // run main function
  blastApp->run();

  printf("BlastUber app finished. Num results: %d\n", outBuff->size());

#if PRINT_OUTPUT_BUFFERS_BLAST
// print contents of output buffer
std::cout << "Output buffer: \n" ;

point* outData = outBuff->get_data();
for(int i=0; i < outBuff->size(); ++i)
   printf("[%d]: (%d, %d)\n", i, outData[i].db, outData[i].query);

#endif

cudaFree(inBufferData);

// NB: pretty-printing currently broken
#if 0

//PRETTY PRINT

        // main pretty-print loop
        std::cout << "*** Formatted results.  Total results: " 
          << outBuff->size() << std::endl;
	for(int j = 0; j < outBuff->size(); ++j) {  
          auto nextPoint = outData[j];
	std::cout << "J = " << j << std::endl;
	int tmpScore = 0;
	int highestL = 0;
	int highestR = 0;
	int tmpr = 8;
	int tmpl = 0;
	int iMin = min(nextPoint.db, nextPoint.query / 4);
	int iMax = min(dSize - nextPoint.db, (qSize - nextPoint.query + 3) / 4);
	const Base* queryy = q + STRING_BUFF * 4 + nextPoint.query;
	const Base* dbb = d + STRING_BUFF + nextPoint.db;
	int mask;
	for(int i = 1; i <= iMin; ++i) {
		int qbyte = queryy[i * -4 + 3];
		int dbyte = dbb[-i];

		for (int k = 0; k < 4; ++k) {
			mask = 0x03 << (2 * k);
			tmpScore += (((qbyte & 0x06) >> 1) == ((dbyte & mask) >> (2 * k)) ? MATCH_SCORE : MISMATCH_SCORE);
			qbyte = queryy[i * -4 + 3 - k - 1];
			if(tmpScore > highestL) {
				highestL = tmpScore;
				tmpl = i - STRING_BUFF;
			}
		}
		if(highestL + tmpScore < MAX_DIFF) {
			break;
		}
	}
	tmpScore = 0;

	for(int i = 2; i <= iMax; ++i) {

		int qbyte = query[i * 4];
		int dbyte = db[i];
      
		for (int k = 0; k < 4; ++k) {
			mask = 0xC0 >> (2 * k);
			tmpScore += (((qbyte & 0x06) << 5) == ((dbyte & mask) << (2 * k)) ? MATCH_SCORE : MISMATCH_SCORE);
			qbyte = query[i * 4 + k + 1];
			if(tmpScore > highestL) {
				highestR = tmpScore;
				tmpr = i + 2 - STRING_BUFF;
			}
		}
		if(highestR + tmpScore < MAX_DIFF) {
			break;
		}
	}

	//Printing Matches
	std::string qTmp = query.substr(nextPoint.query - tmpl * 4, tmpl * 4 + 8 + tmpr * 4);
	std::string dTmp = "";
	//std::cout << "HE" << std::endl;
	/*
	for(int k = nextPoint.query - tmpl; k < nextPoint.query + tmpr; ++k) {
		assert(k > 0);
		assert(k <= query.size());
		char curr = query.at(k);
		switch(curr & 0xC0) {
			case 0xC0:
				//G
				qTmp += "G";
				break;
			case 0x80:
				//T
				qTmp += "T";
				break;
			case 0x40:
				//C
				qTmp += "C";
				break;
			case 0x00:
				//A
				qTmp += "A";
				break;
			default:
				qTmp += "X";
				break;
		};
		switch(curr & 0x30) {
			case 0x30:
				//G
				qTmp += "G";
				break;
			case 0x20:
				//T
				qTmp += "T";
				break;
			case 0x10:
				//C
				qTmp += "C";
				break;
			case 0x00:
				//A
				qTmp += "A";
				break;
			default:
				qTmp += "X";
				break;
		};
		switch(curr & 0x0C) {
			case 0x0C:
				//G
				qTmp += "G";
				break;
			case 0x08:
				//T
				qTmp += "T";
				break;
			case 0x04:
				//C
				qTmp += "C";
				break;
			case 0x00:
				//A
				qTmp += "A";
				break;
			default:
				qTmp += "X";
				break;
		};
		switch(curr & 0x03) {
			case 0x03:
				//G
				qTmp += "G";
				break;
			case 0x02:
				//T
				qTmp += "T";
				break;
			case 0x01:
				//C
				qTmp += "C";
				break;
			case 0x00:
				//A
				qTmp += "A";
				break;
			default:
				qTmp += "X";
				break;
		};
	}
	*/
	//qTmp = qTmp.substr((tmpl + 1) % 4 , qTmp.size() - ((tmpl + 1) / 4) - ((tmpr + 3) / 4));
	//qTmp = qTmp.substr(tmpl % 4, tmpr - tmpl + 9);
	//std::cout << qTmp << " " << qTmp.size() << std::endl;
	//std::string rsp;
	//std::cin >> rsp;
	for(int k = nextPoint.db - tmpl; k < nextPoint.db + tmpr + 2; ++k) {
		assert(k > 0);
		assert(k <= db.size());
		char curr = db.at(k);
		switch(curr & 0xC0) {
			case 0xC0:
				//G
				dTmp += "G";
				break;
			case 0x80:
				//T
				dTmp += "T";
				break;
			case 0x40:
				//C
				dTmp += "C";
				break;
			case 0x00:
				//A
				dTmp += "A";
				break;
			default:
				dTmp += "X";
				break;
		};
		switch(curr & 0x30) {
			case 0x30:
				//G
				dTmp += "G";
				break;
			case 0x20:
				//T
				dTmp += "T";
				break;
			case 0x10:
				//C
				dTmp += "C";
				break;
			case 0x00:
				//A
				dTmp += "A";
				break;
			default:
				dTmp += "X";
				break;
		};
		switch(curr & 0x0C) {
			case 0x0C:
				//G
				dTmp += "G";
				break;
			case 0x08:
				//T
				dTmp += "T";
				break;
			case 0x04:
				//C
				dTmp += "C";
				break;
			case 0x00:
				//A
				dTmp += "A";
				break;
			default:
				dTmp += "X";
				break;
		};
		switch(curr & 0x03) {
			case 0x03:
				//G
				dTmp += "G";
				break;
			case 0x02:
				//T
				dTmp += "T";
				break;
			case 0x01:
				//C
				dTmp += "C";
				break;
			case 0x00:
				//A
				dTmp += "A";
				break;
			default:
				dTmp += "X";
				break;
		};
	}
	//dTmp = dTmp.substr((tmpl + 1) % 4, dTmp.size() - ((tmpl + 1) / 4) - ((tmpr + 3) / 4));
	//dTmp = dTmp.substr(tmpl % 4, tmpr - tmpl + 9);
	//if(tmpl % 4 == 3)
	//	tmpl -= 2;
	//else if(tmpl % 4 == 1)
	//	tmpl += 2;
	std::cout << "DTMP = " << dTmp << "  " << dTmp.size() << std::endl;
	std::cout << "QTMP = " << qTmp << "  " << qTmp.size() << std::endl;
	std::cout << "--------------------------------------------" << std::endl;
	std::cout << nextPoint.query - tmpl << "\t" << std::hex << nextPoint.query - tmpl << std::dec << "\t" << qTmp << "\t" << nextPoint.query + tmpr << std::hex << "\t" << nextPoint.query + tmpr << std::dec << std::endl << "\t\t";
	for(int h = 0; h < qTmp.size(); ++h) {
		//assert(qTmp.size() == dTmp.size());
		//std::cout << "HERE";
		std::cout << (qTmp.at(h) == dTmp.at(h) ? "|" : " ");
	}
	std::cout << std::endl << nextPoint.db * 4 - tmpl * 4 << "\t" << std::hex << nextPoint.db - tmpl << std::dec << "\t" << dTmp << "\t" << nextPoint.db * 4 + tmpr * 4 << std::hex << "\t" << nextPoint.db + tmpr << std::dec << std::endl;
	std::cout << std::dec;
	} // end main pretty-print loop

#endif

// cleanup
cudaFree(blastApp);

}

#endif
