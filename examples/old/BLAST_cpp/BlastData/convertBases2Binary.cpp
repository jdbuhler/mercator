// From B.Y and WaltP on DaniWeb
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int main(int argc, char* argv[]) {
  //char sum = 0;
  char x;
  ifstream inFile(argv[1]);
  if (!inFile) {
    cout << "Unable to open file";
    return 1;
//    exit(1);
  }

  string line ;
  unsigned char nextWord;
  // position of character in next 8-bit word to be built
  // NB: positions are indexed from high-order 2bits to low
  int posInNextWord = 0;
  while( getline( inFile, line ) )
  {
    for (int i=0; i < line.length(); i++)
    {
      if(posInNextWord == 0)  // start over
        nextWord = 0;
      switch(line[i])
      {
        case 'A':
          break;
        case 'C':
          nextWord <<= 2;
          nextWord |= 0x1;
          break;
        case 'G':
          nextWord <<= 2;
          nextWord |= 0x2;
          break;
#if 1
        case 'T':
          nextWord <<= 2;
          nextWord |= 0x3;
          break;
#endif
        default:        // make non-strict-DNA characters 'A'
          break;
      }

      if(posInNextWord == 3)  // finish up: print char
        cout << nextWord;


        // increment char counter
        posInNextWord = (posInNextWord + 1) % 4;
    }
  }

#if 0
  while (x != '\n') {
    while (inFile >> x) {    

      cout<<"x = "<<x<<endl;

    }
  }
#endif

  inFile.close();

  //  getchar();
  return 0;
}
