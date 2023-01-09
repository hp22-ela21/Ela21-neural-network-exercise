/********************************************************************************
* ann.hpp: Innehåller funktionalitet för implementering av artificiella
*          neurala nätverk via klassen ann (ANN = Artificial Neural Network).
********************************************************************************/
#ifndef ANN_HPP_
#define ANN_HPP_

/* Inkluderingsdirektiv: */
#include "dense_layer.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>

/********************************************************************************
* ann: Klass för implementering av neuralt nätverk innehållande ett ingångslager,
*      ett dolt lager samt ett utgångslager med godtyckligt antal noder.
*      Träningsdata kan passeras via vektorer. Efter träning kan prediktion
*      med utskrift genomföras med godtycklig indata eller med indata från 
*      befintliga träningsuppsätningar.
********************************************************************************/
class ann
{
private:
   dense_layer hidden_layer_;                   /* Dolt lager. */
   dense_layer output_layer_;                   /* Utgångslager. */
   std::vector<std::vector<double>> train_in_;  /* Träningsdata in (insignaler). */
   std::vector<std::vector<double>> train_out_; /* Träningsdata ut (referensvärden). */
   std::vector<std::size_t> train_order_;       /* Lagrar ordningsföljden för träningsdatan. */

   /********************************************************************************
   * check_training_data_size: Kontrollerar så att antalet träningsuppsättningar
   *                           med indata är samma som antalet träningsuppsättningar
   *                           med utdata. Om detta inte är fallet kortas den
   *                           större träningsuppsättningen av så att den matchar
   *                           den mindre uppsättningen.
   ********************************************************************************/
   void check_training_data_size(void)
   {
      return;
   }

   /********************************************************************************
   * init_training_order: Initierar vektor innehållande ordningsföljden för
   *                      träningsuppsättningarna. Vektorns storlek sätts till
   *                      antalet träningsuppsättningar och den tilldelas index
   *                      i stigande ordning från 0.
   ********************************************************************************/
   void init_training_order(void)
   {
      return;
   }

   /********************************************************************************
   * feedforward: Beräknar nya utsignaler för samtliga noder i det neurala nätverk
   *              via angiven indata.
   * 
   *              - input: Referens till vektor innehållande ny indata.
   ********************************************************************************/
   void feedforward(const std::vector<double>& input)
   {
      return;
   }

   /********************************************************************************
   * backpropagate: Beräknar aktuella fel för samtliga noder i angiven neuralt
   *                nätverk via jämförelse med referensdata innehållande korrekta
   *                utsignaler, vilket jämförs med predikterade utsignaler.
   *
   *                - reference: Referens till vektor innehållande korrekta värden.
   ********************************************************************************/
   void backpropagate(const std::vector<double>& reference)
   {
      return;
   }

   /********************************************************************************
   * optimize: Justerar parametrar i angivet neuralt nätverk för att minska 
   *           uppkommet fel. Vid nästa prediktion bör därmed felet ha minskat
   *           och precisionen är då högre.
   *
   *           - input        : Referens till vektor innehållande aktuell indata.
   *           - learning_rate: Lärhastigheten, avgör justeringsgraden av
   *                            parametrarna vid fel.
   ********************************************************************************/
   void optimize(const std::vector<double>& input,
                 const double learning_rate)
   {
      return;
   }

   /********************************************************************************
   * randomize_training_order: Randomiserar ordningsföljden för befintliga 
   *                           träningsuppsättningar i angivet neuralt nätverk.
   *                           I praktiken flyttas innehållet på index i samt
   *                           randomiserat index r.
   ********************************************************************************/
   void randomize_training_order(void)
   {
      return;
   }

public:

   /********************************************************************************
   * ann: Defaultkonstruktor, initierar ett tomt neuralt nätverk.
   ********************************************************************************/
   ann(void) { }

   /********************************************************************************
   * ann: Initierar neuralt nätverk med angivet antal noder i respektive lager.
   *
   *      - num_inputs      : Antalet noder i ingångslagret (antalet insignaler).
   *      - num_hidden_nodes: Antalet noder i det dolda lagret.
   *      - num_outputs     : Antalet noder i utgångslagret (antalet utsignaler).
   ********************************************************************************/
   ann(const std::size_t num_inputs,
       const std::size_t num_hidden_nodes,
       const std::size_t num_outputs)
   {
      this->init(num_inputs, num_hidden_nodes, num_outputs);
      return;
   }

   /********************************************************************************
   * ~ann: Destruktor, tömmer neuralt nätverk automatiskt när det går ur scope.
   ********************************************************************************/
   ~ann(void)
   {
      this->clear();
      return;
   }

   /********************************************************************************
   * hidden_layer: Returnerar en referens till det dolda lagret i angivet neuralt
   *               nätverk så att användaren kan läsa innehållet, men inte skriva.
   ********************************************************************************/
   const dense_layer& hidden_layer(void) const
   {
      return this->hidden_layer_;
   }

   /********************************************************************************
   * output_layer: Returnerar en referens till utgångslagret i angivet neuralt
   *               nätverk så att användaren kan läsa innehållet, men inte skriva.
   ********************************************************************************/
   const dense_layer& output_layer(void) const
   {
      return this->output_layer_;
   }

   /********************************************************************************
   * train_in: Returnerar en referens till en vektor innehållande träningsdata
   *           bestående av insignaler.
   ********************************************************************************/
   const std::vector<std::vector<double>>& train_in(void) const
   {
      return this->train_in_;
   }

   /********************************************************************************
   * train_out: Returnerar en referens till en vektor innehållande träningsdata
   *            bestående av utsignaler.
   ********************************************************************************/
   const std::vector<std::vector<double>>& train_out(void) const
   {
      return this->train_out_;
   }

   /********************************************************************************
   * num_inputs: Returnerar antalet ingångsnoder i angivet neuralt nätverk, vilket
   *             är samma som antalet vikter per nod i det dolda lagret.
   ********************************************************************************/
   std::size_t num_inputs(void) const
   {
      return this->hidden_layer_.num_weights();
   }

   /********************************************************************************
   * num_hidden_nodes: Returnerar antalet noder i det dolda lagret i angivet
   *                   neuralt nätverk.
   ********************************************************************************/
   std::size_t num_hidden_nodes(void) const
   {
      return this->hidden_layer_.num_nodes();
   }

   /********************************************************************************
   * num_outputs: Returnerar antalet utgångsnoder i angivet neuralt nätverk.
   ********************************************************************************/
   std::size_t num_outputs(void) const
   {
      return this->output_layer_.num_nodes();
   }

   /********************************************************************************
   * num_training_sets: Returnerar antalet befintliga träningsuppsättningar i
   *                    angivet neuralt nätverk.
   ********************************************************************************/
   std::size_t num_training_sets(void) const
   {
      return this->train_order_.size();
   }

   /********************************************************************************
   * output: Returnerar en referens till utsignalerna i utgångslagret på angivet
   *         neuralt nätverk.
   ********************************************************************************/
   const std::vector<double>& output(void) const
   {
      return this->output_layer_.output;
   }

   /********************************************************************************
   * init: Initierar neuralt nätverk med angivet antal noder i respektive lager.
   * 
   *       - num_inputs      : Antalet noder i ingångslagret (antalet insignaler).
   *       - num_hidden_nodes: Antalet noder i det dolda lagret.
   *       - num_outputs     : Antalet noder i utgångslagret (antalet utsignaler).
   ********************************************************************************/
   void init(const std::size_t num_inputs,
             const std::size_t num_hidden_nodes,
             const std::size_t num_outputs)
   {
       return;
   }

   /********************************************************************************
   * clear: Tömmer angivet neuralt nätverk.
   ********************************************************************************/
   void clear(void)
   {
      return;
   }

   /********************************************************************************
   * set_training_data: Lagrar träningsdata för angivet neuralt nätverk via 
   *                    kopiering av innehållet från refererade vektorer.
   *                    Ifall ett ojämnt antal träningsuppsättningar passeras,
   *                    exempelvis sju för indata och fem för utdata, sparas
   *                    endast antalet befintliga träningsuppsättningar som består
   *                    av både in- och utdata, alltså fem i ovanstående exempel.
   * 
   *                    - train_in : Referens till vektor innehållande indata.
   *                    - train_out: Referens till vektor innehållande utdata.
   ********************************************************************************/
   void set_training_data(const std::vector<std::vector<double>>& train_in,
                          const std::vector<std::vector<double>>& train_out)
   {
      return;
   }

   /********************************************************************************
   * train: Tränar angivet neuralt nätverk under angivet antal epoker med 
   *        godtycklig lärhastighet.
   * 
   *        - num_epochs   : Antalet epoker som ska träning ska genomföras under.
   *        - learning_rate: Lärhastigheten, avgör hur mycket nätverkets parametrar
   *                         justeras vid fel.
   ********************************************************************************/
   void train(const std::size_t num_epochs,
              const double learning_rate)
   {
      return;
   }

   /********************************************************************************
   * predict: Genomför prediktion via angiven indata och returnerar en referens
   *          till en vektor innehållande utdatan.
   * 
   *          - input: Referens till vektor innehållande indata.
   ********************************************************************************/
   const std::vector<double>& predict(const std::vector<double>& input)
   {
      this->feedforward(input);
      return this->output();
   }

   /********************************************************************************
   * print: Genomför prediktion med indata från angiven vektor och skriver ut 
   *        predikterad utdata via angiven utström, där standardutenheten std::cout 
   *        används som default för utskrift i terminalen. Som default sker utskrift 
   *        med en decimal. Värden mellan -0.001 samt 0.001 avrundas till noll vid 
   *        utskrift.
   *
   *        - input       : Referens till vektor innehållande indata.
   *        - num_decimals: Antalet decimaler vid utskrift (default = 1).
   *        - ostream     : Referens till godtycklig utström (default = std::cout).
   *        - threshold   : Tröskelvärde som används för att avrunda tal nära
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
   * print: Genomför prediktion med samtliga befintliga träningsuppsättningars
   *        indata och skriver ut predikterad utdata via angiven utström, där
   *        standardutenheten std::cout används som default för utskrift i 
   *        terminalen. Som default sker utskrift med en decimal. Värden mellan
   *        -0.001 samt 0.001 avrundas till noll vid utskrift.
   * 
   *        - num_decimals: Antalet decimaler vid utskrift (default = 1).
   *        - ostream     : Referens till godtycklig utström (default = std::cout).
   *        - threshold   : Tröskelvärde som används för att avrunda tal nära
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