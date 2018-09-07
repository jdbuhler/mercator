#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <math.h>
#include <limits.h>

using namespace std;

const int KMER_SIZE = 8;	//KMER_SIZE is 1 less than actual size, due to 0 base index vectors
const int DIAG_DIST_MAX = 1;	//Maximum diagonal distance used for calculating kmer overlap in diagonalFilter
//const int UNGAP_DIST_MAX = 64;	//Maximum ungapped extension distance for calculating kmer matching scores in ungapExt
const int UNGAP_DIST_MAX = 8;	//Maximum ungapped extension distance for calculating kmer matching scores in ungapExt
//const int THRESH = 2;		//Minimum score threshold that must be achieved for kmer match to be kept in ungapExt
const int THRESH = 3;		//Minimum score threshold that must be achieved for kmer match to be kept in ungapExt
const int MISMATCH_SCORE = -3;	//Score added to the total score in ungapExt when a mismatch occurs
const int MATCH_SCORE = 1;	//Score added to the total score in ungapExt when a match occurs

#define UNCOMPRESSED_PRINT 0	//Defines whether the final printing is compressed to query and database position and match count, or full printing.
#define PRINT_KMERS 0	//Defines whether to print all of the kmers found in the queryHashes.

bool ungapExt(int x, int y, string query, string db);
void prettyPrint(int x, int y, string query, string db);

typedef unordered_map<string, vector<int> > hashtable;

int main() {

	//File I/O
        // NB: Both files should be in ASCII format, not compressed format
//	string queryFileName = "query.txt";
//	string dbFileName = "db.txt";
	string queryFileName = "./BlastData/salmonella-2k.txt";
	string dbFileName = "./BlastData/ecoli-k12-noheader.txt";
	
#if 0
	string query;
	string db;	

	ifstream queryFile;
	queryFile.open(queryFilename.c_str());
	//queryFile >> query;
	queryFile >> query;
	queryFile.close();
	
	ifstream dbFile;
	dbFile.open(dbFilename.c_str());
	//dbFile >> db;
	dbFile >> db;
	dbFile.close();
#endif

/////////////////////
	std::string db = "";
	std::string query = "";
	std::string tmp = "";

	std::ifstream queryFile;
//        std::string queryFileName = QUERYFILENAME;
	queryFile.open(queryFileName.c_str(), std::ifstream::binary);

	std::ifstream dbFile;
//        std::string dbFileName = DBFILENAME;
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
/////////////////////


	hashtable queryHashes;	//The hash table containing the kmer positions in the query.

	vector< pair<int, int> > hits = vector< pair<int, int> >();	//Final hits after seed match and diagonal filter.
	int* diagonal = new int[query.size()];	//Array for diagonal calculations.
	
	//Fill diagonal with default values of INT_MIN.
	for(int i = 0; i < query.size(); ++i) {
		diagonal[i] = INT_MIN;
	}
	
	//Construct the queryHashes hashtable with the given query string.
	for(int i = 0; i < query.size() - KMER_SIZE; ++i) {
		auto a = queryHashes.insert(make_pair(query.substr(i, KMER_SIZE + 1), vector<int>()));
		a.first->second.push_back(i);

		#if PRINT_KMERS
		hashtable::const_iterator b = queryHashes.find(query.substr(i, KMER_SIZE + 1));
		cout << b->first << endl;
		for(int j = 0; j < b->second.size(); ++j)
			cout << b->second.at(j) << " ";
		cout << endl;
		#endif
	}

        ////////// coles start timing here
	
	//SEED MATCHING AND DIAGONAL FILTERING
	for(int i = 0; i < db.size() - KMER_SIZE; ++i) {
		//SEED MATCH
		hashtable::iterator a = queryHashes.find(db.substr(i, KMER_SIZE + 1));

		//IMMEDIATELY RUN DIAGONAL FILTERING
		if(a != queryHashes.end()) {
			if(a->second.size() != 0) {
				//Iterate through each query position the Kmer is located at for the current database position.
				for(int j = 0; j < a->second.size(); ++j) {
					//Set new diagonal start position if the DIAG_DIST_MAX is not reached.
					if(diagonal[(i - a->second.at(j) + query.size()) % query.size()] + DIAG_DIST_MAX <= i) {
						diagonal[(i - a->second.at(j) + query.size()) % query.size()] = i;
						hits.push_back(pair<int, int>(a->second.at(j), i));
					}
				}
			}
		}
		//Reset the next query position in the diagonal array.
		diagonal[(i + query.size() + 1) % query.size()] = INT_MIN;
	}
	
	//Perform ungapped extension on the hits passed through by the seed matching and diagonal filtering.
	//Does not guarantee hits will be ordered after processing.
	for(int i = 0; i < hits.size(); ++i) {
		if(ungapExt(hits.at(i).first, hits.at(i).second, query, db) == false)  {
			pair<int, int> tmp = pair<int, int>(hits.at(hits.size() - 1).first, hits.at(hits.size() - 1).second);
			hits.at(i) = tmp;
			hits.pop_back();
			--i;
		}				
	}

        /////// coles stop timing here

	//Print the remaining matches to the console.
	//Currently prints in order, but can be printed out of order in linear time.
        cout << "Results: " << hits.size() << " total.\n";
	for(int j = 0; j < query.size() - KMER_SIZE; ++j) {
		for(int i = 0; i < hits.size(); ++i) {
			if(hits.at(i).first == j) {
//				if(hits.at(i).first >= 67000 && hits.at(i).first <= 67100)
				prettyPrint(hits.at(i).first, hits.at(i).second, query, db);
			}
		}
	}

	return 0;
}

/*
 * Runs a fixed ungapped extension on a kmer at position x, y, and returns true if
 * the maximum score of matches is above THRESH, otherwise false.
 * @param x The x starting position in the matrix of the current kmer
 * @param y The y starting position in the matrix of the current kmer
 * @param query The query string searching for
 * @param db The database string searching through
 * @return bool True if score of highest extension right and left plus KMER_SIZE is greater than THRESH, else false
 */
bool ungapExt(int x, int y, string query, string db) {
	int highestL = 0;
	int highestR = 0;
	int tempScore = 0;
	int iMin = min(x, y);
	iMin = -1 * min(iMin, UNGAP_DIST_MAX - 1);
	int iMax = min(db.size() - (y + KMER_SIZE + 1), query.size() - (x + KMER_SIZE + 1));
	iMax = min(iMax, UNGAP_DIST_MAX);

	//Check left of Kmer	
	//for(int i = -1; i > iMin && x >= -1 * i && y >= -1 * i; --i) {
	for(int i = -1; i >= iMin; --i) {
		if(query.at(x + i) == db.at(y + i)) {
			tempScore += MATCH_SCORE;
		}		
		else {
			tempScore += MISMATCH_SCORE;
		}
		if(tempScore > highestL) {
			highestL = tempScore;
		}
	}

	tempScore = 0;

	//Check right of Kmer
	//for(int i = 1; i < UNGAP_DIST_MAX && y + i + KMER_SIZE + 1 < db.size() && x + i + KMER_SIZE + 1 < query.size(); ++i) {
	for(int i = 1; i < iMax; ++i) {
		if(query.at(x + i + KMER_SIZE) == db.at(y + i + KMER_SIZE)) {
			tempScore += MATCH_SCORE;
		}		
		else {
			tempScore += MISMATCH_SCORE;
		}
		if(tempScore > highestR) {
			highestR = tempScore;
		}		
	}
	return highestR + highestL + KMER_SIZE + 1 > THRESH;
}

/*
 * Prints out the maximum scoring substring of the provided kmer position, along with
 * the given x, y position, as well as the match's length.
 * @param x The x starting position in the matrix of the current kmer (query)
 * @param y The y starting position in the matrix of the current kmer (database)
 * @param query The query string searching for
 * @param db The database string searching through
 */
void prettyPrint(int x, int y, string query, string db) {
	int highestL = 0;
	int highestR = 0;
	int tmpl = 0;
	int tmpr = KMER_SIZE;
	int tempScore = 0;
	int iMin = min(x, y);
	iMin = -1 * min(iMin, UNGAP_DIST_MAX - 1);
	int iMax = min(db.size() - (y + KMER_SIZE + 1), query.size() - (x + KMER_SIZE + 1));
	iMax = min(iMax, UNGAP_DIST_MAX);

	//Check left of Kmer	
	//for(int i = -1; i > -1 * UNGAP_DIST_MAX && x + i >= 0 && y + i >= 0; --i) {
	for(int i = -1; i >= iMin; --i) {
		if(query.at(x + i) == db.at(y + i)) {
			tempScore += MATCH_SCORE;	
		}		
		else {
			tempScore += MISMATCH_SCORE;
		}
		if(tempScore > highestL) {
			highestL = tempScore;
			tmpl = i;
		}
	}

	tempScore = 0;

	//Check right of Kmer
	//for(int i = 1; i < UNGAP_DIST_MAX && y + i + KMER_SIZE + 1 < db.size() && x + i + KMER_SIZE + 1 < query.size(); ++i) {
	for(int i = 1; i < iMax; ++i) {
		if(query.at(x + i + KMER_SIZE) == db.at(y + i + KMER_SIZE)) {
			tempScore += MATCH_SCORE;	
		}		
		else {
			tempScore += MISMATCH_SCORE;
		}
		if(tempScore > highestR) {
			highestR = tempScore;
			tmpr = i + KMER_SIZE;
		}			
	}

	//Print the matches along with the positions in the database and query
	//including string length.
	string q = query.substr(x + tmpl, tmpr - tmpl + 1);
	string d = db.substr(y + tmpl, tmpr - tmpl + 1);
	cout << x << "  " << y << "  " << tmpr - tmpl + 1 << endl;
	#if UNCOMPRESSED_PRINT
	cout << q << endl;
	for(int i = 0; i < q.size(); ++i) {
		if(q.at(i) == d.at(i))
			cout << "|";
		else
			cout << " ";
	}
	cout << endl << d << endl;
	cout << "------------------------------------------" << endl;
	#endif
}

