/********************************************************************************
* main.cpp: Implementering av ett enkelt neuralt n�tverk i C++.
********************************************************************************/
#include "ann.hpp"
#include <vector>

/********************************************************************************
* main: Skapar ett neuralt n�tverk inneh�llande tv� ing�ngsnoder, tv� noder
*       i det enda ing�ngslagret och tv� utg�ngsnoder. N�tverk tr�nas f�r att 
*       detektera ett 2-ing�ngars XOR-m�nster under 1000 epoker med en 
*       l�rhastighet p� 2 %. Efter tr�ningen genomf�rs prediktion samt utskrift
*       med samtliga tr�ningsupps�ttningars insignaler som indata.
* 
*       XOR-m�nstret f�r insignaler X1 och X2 samt utsignal Y visas nedan:
*     
*       X1 X2 Y
*       0  0  0
*       0  1  1
*       1  0  1
*       1  1  0
********************************************************************************/
int main(void)
{
   const std::vector<std::vector<double>> train_in = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
   const std::vector<std::vector<double>> train_out = { { 0 }, { 1 }, { 1 }, { 0 } };

   ann ann1(2, 2, 1);
   ann1.set_training_data(train_in, train_out);
   ann1.train(1000, 0.02);
   ann1.print();
   return 0;
}