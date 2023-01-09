/********************************************************************************
* ann.hpp: Inneh�ller funktionalitet f�r implementering av artificiella
*          neurala n�tverk via klassen ann (ANN = Artificial Neural Network).
********************************************************************************/
#ifndef ANN_HPP_
#define ANN_HPP_

/* Inkluderingsdirektiv: */
#include "dense_layer.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>

/********************************************************************************
* ann: Klass f�r implementering av neuralt n�tverk inneh�llande ett ing�ngslager,
*      ett dolt lager samt ett utg�ngslager med godtyckligt antal noder.
*      Tr�ningsdata kan passeras via vektorer. Efter tr�ning kan prediktion
*      med utskrift genomf�ras med godtycklig indata eller med indata fr�n 
*      befintliga tr�ningsupps�tningar.
********************************************************************************/
class ann
{
private:
   dense_layer hidden_layer_;                   /* Dolt lager. */
   dense_layer output_layer_;                   /* Utg�ngslager. */
   std::vector<std::vector<double>> train_in_;  /* Tr�ningsdata in (insignaler). */
   std::vector<std::vector<double>> train_out_; /* Tr�ningsdata ut (referensv�rden). */
   std::vector<std::size_t> train_order_;       /* Lagrar ordningsf�ljden f�r tr�ningsdatan. */

   /********************************************************************************
   * check_training_data_size: Kontrollerar s� att antalet tr�ningsupps�ttningar
   *                           med indata �r samma som antalet tr�ningsupps�ttningar
   *                           med utdata. Om detta inte �r fallet kortas den
   *                           st�rre tr�ningsupps�ttningen av s� att den matchar
   *                           den mindre upps�ttningen.
   ********************************************************************************/
   void check_training_data_size(void)
   {
      return;
   }

   /********************************************************************************
   * init_training_order: Initierar vektor inneh�llande ordningsf�ljden f�r
   *                      tr�ningsupps�ttningarna. Vektorns storlek s�tts till
   *                      antalet tr�ningsupps�ttningar och den tilldelas index
   *                      i stigande ordning fr�n 0.
   ********************************************************************************/
   void init_training_order(void)
   {
      return;
   }

   /********************************************************************************
   * feedforward: Ber�knar nya utsignaler f�r samtliga noder i det neurala n�tverk
   *              via angiven indata.
   * 
   *              - input: Referens till vektor inneh�llande ny indata.
   ********************************************************************************/
   void feedforward(const std::vector<double>& input)
   {
      return;
   }

   /********************************************************************************
   * backpropagate: Ber�knar aktuella fel f�r samtliga noder i angiven neuralt
   *                n�tverk via j�mf�relse med referensdata inneh�llande korrekta
   *                utsignaler, vilket j�mf�rs med predikterade utsignaler.
   *
   *                - reference: Referens till vektor inneh�llande korrekta v�rden.
   ********************************************************************************/
   void backpropagate(const std::vector<double>& reference)
   {
      return;
   }

   /********************************************************************************
   * optimize: Justerar parametrar i angivet neuralt n�tverk f�r att minska 
   *           uppkommet fel. Vid n�sta prediktion b�r d�rmed felet ha minskat
   *           och precisionen �r d� h�gre.
   *
   *           - input        : Referens till vektor inneh�llande aktuell indata.
   *           - learning_rate: L�rhastigheten, avg�r justeringsgraden av
   *                            parametrarna vid fel.
   ********************************************************************************/
   void optimize(const std::vector<double>& input,
                 const double learning_rate)
   {
      return;
   }

   /********************************************************************************
   * randomize_training_order: Randomiserar ordningsf�ljden f�r befintliga 
   *                           tr�ningsupps�ttningar i angivet neuralt n�tverk.
   *                           I praktiken flyttas inneh�llet p� index i samt
   *                           randomiserat index r.
   ********************************************************************************/
   void randomize_training_order(void)
   {
      return;
   }

public:

   /********************************************************************************
   * ann: Defaultkonstruktor, initierar ett tomt neuralt n�tverk.
   ********************************************************************************/
   ann(void) { }

   /********************************************************************************
   * ann: Initierar neuralt n�tverk med angivet antal noder i respektive lager.
   *
   *      - num_inputs      : Antalet noder i ing�ngslagret (antalet insignaler).
   *      - num_hidden_nodes: Antalet noder i det dolda lagret.
   *      - num_outputs     : Antalet noder i utg�ngslagret (antalet utsignaler).
   ********************************************************************************/
   ann(const std::size_t num_inputs,
       const std::size_t num_hidden_nodes,
       const std::size_t num_outputs)
   {
      this->init(num_inputs, num_hidden_nodes, num_outputs);
      return;
   }

   /********************************************************************************
   * ~ann: Destruktor, t�mmer neuralt n�tverk automatiskt n�r det g�r ur scope.
   ********************************************************************************/
   ~ann(void)
   {
      this->clear();
      return;
   }

   /********************************************************************************
   * hidden_layer: Returnerar en referens till det dolda lagret i angivet neuralt
   *               n�tverk s� att anv�ndaren kan l�sa inneh�llet, men inte skriva.
   ********************************************************************************/
   const dense_layer& hidden_layer(void) const
   {
      return this->hidden_layer_;
   }

   /********************************************************************************
   * output_layer: Returnerar en referens till utg�ngslagret i angivet neuralt
   *               n�tverk s� att anv�ndaren kan l�sa inneh�llet, men inte skriva.
   ********************************************************************************/
   const dense_layer& output_layer(void) const
   {
      return this->output_layer_;
   }

   /********************************************************************************
   * train_in: Returnerar en referens till en vektor inneh�llande tr�ningsdata
   *           best�ende av insignaler.
   ********************************************************************************/
   const std::vector<std::vector<double>>& train_in(void) const
   {
      return this->train_in_;
   }

   /********************************************************************************
   * train_out: Returnerar en referens till en vektor inneh�llande tr�ningsdata
   *            best�ende av utsignaler.
   ********************************************************************************/
   const std::vector<std::vector<double>>& train_out(void) const
   {
      return this->train_out_;
   }

   /********************************************************************************
   * num_inputs: Returnerar antalet ing�ngsnoder i angivet neuralt n�tverk, vilket
   *             �r samma som antalet vikter per nod i det dolda lagret.
   ********************************************************************************/
   std::size_t num_inputs(void) const
   {
      return this->hidden_layer_.num_weights();
   }

   /********************************************************************************
   * num_hidden_nodes: Returnerar antalet noder i det dolda lagret i angivet
   *                   neuralt n�tverk.
   ********************************************************************************/
   std::size_t num_hidden_nodes(void) const
   {
      return this->hidden_layer_.num_nodes();
   }

   /********************************************************************************
   * num_outputs: Returnerar antalet utg�ngsnoder i angivet neuralt n�tverk.
   ********************************************************************************/
   std::size_t num_outputs(void) const
   {
      return this->output_layer_.num_nodes();
   }

   /********************************************************************************
   * num_training_sets: Returnerar antalet befintliga tr�ningsupps�ttningar i
   *                    angivet neuralt n�tverk.
   ********************************************************************************/
   std::size_t num_training_sets(void) const
   {
      return this->train_order_.size();
   }

   /********************************************************************************
   * output: Returnerar en referens till utsignalerna i utg�ngslagret p� angivet
   *         neuralt n�tverk.
   ********************************************************************************/
   const std::vector<double>& output(void) const
   {
      return this->output_layer_.output;
   }

   /********************************************************************************
   * init: Initierar neuralt n�tverk med angivet antal noder i respektive lager.
   * 
   *       - num_inputs      : Antalet noder i ing�ngslagret (antalet insignaler).
   *       - num_hidden_nodes: Antalet noder i det dolda lagret.
   *       - num_outputs     : Antalet noder i utg�ngslagret (antalet utsignaler).
   ********************************************************************************/
   void init(const std::size_t num_inputs,
             const std::size_t num_hidden_nodes,
             const std::size_t num_outputs)
   {
       return;
   }

   /********************************************************************************
   * clear: T�mmer angivet neuralt n�tverk.
   ********************************************************************************/
   void clear(void)
   {
      return;
   }

   /********************************************************************************
   * set_training_data: Lagrar tr�ningsdata f�r angivet neuralt n�tverk via 
   *                    kopiering av inneh�llet fr�n refererade vektorer.
   *                    Ifall ett oj�mnt antal tr�ningsupps�ttningar passeras,
   *                    exempelvis sju f�r indata och fem f�r utdata, sparas
   *                    endast antalet befintliga tr�ningsupps�ttningar som best�r
   *                    av b�de in- och utdata, allts� fem i ovanst�ende exempel.
   * 
   *                    - train_in : Referens till vektor inneh�llande indata.
   *                    - train_out: Referens till vektor inneh�llande utdata.
   ********************************************************************************/
   void set_training_data(const std::vector<std::vector<double>>& train_in,
                          const std::vector<std::vector<double>>& train_out)
   {
      return;
   }

   /********************************************************************************
   * train: Tr�nar angivet neuralt n�tverk under angivet antal epoker med 
   *        godtycklig l�rhastighet.
   * 
   *        - num_epochs   : Antalet epoker som ska tr�ning ska genomf�ras under.
   *        - learning_rate: L�rhastigheten, avg�r hur mycket n�tverkets parametrar
   *                         justeras vid fel.
   ********************************************************************************/
   void train(const std::size_t num_epochs,
              const double learning_rate)
   {
      return;
   }

   /********************************************************************************
   * predict: Genomf�r prediktion via angiven indata och returnerar en referens
   *          till en vektor inneh�llande utdatan.
   * 
   *          - input: Referens till vektor inneh�llande indata.
   ********************************************************************************/
   const std::vector<double>& predict(const std::vector<double>& input)
   {
      this->feedforward(input);
      return this->output();
   }

   /********************************************************************************
   * print: Genomf�r prediktion med indata fr�n angiven vektor och skriver ut 
   *        predikterad utdata via angiven utstr�m, d�r standardutenheten std::cout 
   *        anv�nds som default f�r utskrift i terminalen. Som default sker utskrift 
   *        med en decimal. V�rden mellan -0.001 samt 0.001 avrundas till noll vid 
   *        utskrift.
   *
   *        - input       : Referens till vektor inneh�llande indata.
   *        - num_decimals: Antalet decimaler vid utskrift (default = 1).
   *        - ostream     : Referens till godtycklig utstr�m (default = std::cout).
   *        - threshold   : Tr�skelv�rde som anv�nds f�r att avrunda tal n�ra
   *                        noll (default = 0.001).
   ********************************************************************************/
   void print(const std::vector<std::vector<double>>& input,
              const std::size_t num_decimals = 1,
              std::ostream& ostream = std::cout,
              const double threshold = 0.001)
   {
      const auto& end = input[input.size() - 1];
      if (input.size() == 0) return;
      ostream << "--------------------------------------------------------------------------------\n";
      ostream << "--------------------------------------------------------------------------------\n\n";
      return;
   }

   /********************************************************************************
   * print: Genomf�r prediktion med samtliga befintliga tr�ningsupps�ttningars
   *        indata och skriver ut predikterad utdata via angiven utstr�m, d�r
   *        standardutenheten std::cout anv�nds som default f�r utskrift i 
   *        terminalen. Som default sker utskrift med en decimal. V�rden mellan
   *        -0.001 samt 0.001 avrundas till noll vid utskrift.
   * 
   *        - num_decimals: Antalet decimaler vid utskrift (default = 1).
   *        - ostream     : Referens till godtycklig utstr�m (default = std::cout).
   *        - threshold   : Tr�skelv�rde som anv�nds f�r att avrunda tal n�ra
   *                        noll (default = 0.001).
   ********************************************************************************/
   void print(const std::size_t num_decimals = 1,
              std::ostream& ostream = std::cout,
              const double threshold = 0.001)
   {
      this->print(this->train_in_, num_decimals, ostream, threshold);
      return;
   }
};


#endif /* ANN_HPP_ */