/********************************************************************************
* dense_layer.hpp: Inneh�ller funktionalitet f�r implementering av dense-lager
*                  i neurala n�tverk via strukten dense_layer.
********************************************************************************/
#ifndef DENSE_LAYER_HPP_
#define DENSE_LAYER_HPP_

/* Inkluderingsdirektiv: */
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>

/********************************************************************************
* dense_layer: Strukt f�r implementering av dense-lager med valbart antal noder
*              samt vikter per nod i neurala n�tverk. Bias och vikter f�r
*              samtliga noder erh�ller randomiserade startv�rden mellan 0 - 1,
*              �vriga parametrar s�tts till 0 vid start.
********************************************************************************/
struct dense_layer
{
   std::vector<double> output;               /* Nodernas utsignaler. */
   std::vector<double> error;                /* Nodernas uppm�tta fel/avvikelser. */
   std::vector<double> bias;                 /* Nodernas vilov�rden (m-v�rden). */
   std::vector<std::vector<double>> weights; /* Nodernas vikter (k-v�rden). */

   /********************************************************************************
   * dense_layer: Initierar nytt tomt dense-lager.
   ********************************************************************************/
   dense_layer(void) { }

   /********************************************************************************
   * dense_layer: Initierar nytt dense-lager av angiven storlek.
   * 
   *              - num_nodes  : Antalet noder i det nya dense-lagret.
   *              - num_weights: Antalet vikter per nod i det nya dense-lagret.
   ********************************************************************************/
   dense_layer(const std::size_t num_nodes,
               const std::size_t num_weights)
   {
      this->resize(num_nodes, num_weights);
      return;
   }

   /********************************************************************************
   * ~dense_layer: T�mmer angivet dense-lager om detta lager g�r ur scope,
   *               exempelvis om ett lokalt dense-lager har deklarerats och
   *               funktionen d�r detta lager har deklarerats avslutas (lagret
   *               frig�rs d� automatiskt fr�n stacken).
   ********************************************************************************/
   ~dense_layer(void)
   {
      this->clear();
      return;
   }

   /********************************************************************************
   * num_nodes: Returnerar antalet befintliga noder i angivet dense-lager.
   ********************************************************************************/
   inline std::size_t num_nodes(void) const
   {
      return this->output.size();
   }

   /********************************************************************************
   * num_weights: Returnerar antalet vikter per nod i angivet dense-lager.
   ********************************************************************************/
   inline std::size_t num_weights(void) const
   {
      return this->weights.size() ? this->weights[0].size() : 0;
   }

   /********************************************************************************
   * clear: T�mmer angiven vektor.
   ********************************************************************************/
   void clear(void)
   {
      this->output.clear();
      this->error.clear();
      this->bias.clear();
      this->weights.clear();
      return;
   }

   /********************************************************************************
   * resize: S�tter antalet noder och vikter per nod i angiven vektor. Bias och
   *         vikter tilldelas randomiserade startv�rden mellan 0 - 1, �vriga
   *         parametrar s�tts till 0 vid start. Innan storleken s�tts s� t�ms
   *         dense-lagret f�r att g�ra initieringen enklare.
   * 
   *         - num_nodes  : Antalet noder i dense-lagret.
   *         - num_weights: Antalet vikter per nod i dense-lagret.
   ********************************************************************************/
   void resize(const std::size_t num_nodes,
               const std::size_t num_weights)
   {
      this->output.resize(num_nodes, 0.0);
      this->error.resize(num_nodes, 0.0);
      this->bias.resize(num_nodes, 0.0);
      this->weights.resize(num_nodes, std::vector<double>(num_weights, 0.0));

      for (std::size_t i = 0; i < num_nodes; ++i)
      {
         this->bias[i] = this->get_random();

         for (std::size_t j = 0; j < num_weights; ++j)
         {
            this->weights[i][j] = this->get_random();
         }
      }

      return;
   }

   /********************************************************************************
   * print: Skriver ut flyttal fr�n angiven vektor p� en rad via angiven utstr�m.
   *
   *        - data   : Referens till vektorn vars inneh�ll ska skrivas ut.
   *        - ostream: Referens till angiven utstr�m.
   ********************************************************************************/
   static void print(const std::vector<double>& data,
                     std::ostream& ostream = std::cout,
                     const std::size_t num_decimals = 1,
                     const double threshold = 0.001)
   {
      ostream << std::fixed;

      for (auto& i : data)
      {
         ostream << std::setprecision(num_decimals) << get_rounded(i, threshold) << " ";
      }

      ostream << "\n";
      return;
   }

   /********************************************************************************
   * print: Skriver ut information om angivet dense-lager via angiven utstr�m,
   *        d�r standardutenheten std::cout anv�nds som default f�r utskrift i 
   *        terminalen.
   * 
   *        - ostream: Referens till angiven utstr�m (default = std::cout).
   ********************************************************************************/
   void print(std::ostream& ostream = std::cout) const
   {
      ostream << "--------------------------------------------------------------------------------\n";
      ostream << "Number of nodes: " << this->num_nodes() << "\n";
      ostream << "Number of weights per node: " << this->num_weights() << "\n\n";

      ostream << "Output: ";
      this->print(this->output, ostream);

      ostream << "Error: ";
      this->print(this->error, ostream);

      ostream << "Bias: ";
      this->print(this->bias, ostream);

      ostream << "\nWeights:\n";

      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         ostream << "Node " << i + 1 << ": ";
         this->print(this->weights[i], ostream);
      }

      ostream << "--------------------------------------------------------------------------------\n\n";
      return;
   }

   /********************************************************************************
   * feedforward: Ber�knar nya utsignaler f�r varje nod i angivet dense-lager
   *              genom att summera respektive nods bias samt indata (vikter *
   *              nya insignaler). Om denna summa �verstiger 0 s� �r noden 
   *              aktiverad och utsignalen s�tts till ber�knad summa. Annars �r 
   *              noden inaktiverad och utsignalen s�tts d� till 0.
   * 
   *              - input: Referens till vektor med nya insignaler.
   ********************************************************************************/
   void feedforward(const std::vector<double>& input)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         auto sum = this->bias[i];

         for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
         {
            sum += input[j] * this->weights[i][j];
         }

         this->output[i] = this->relu(sum);
      }

      return;
   }

   /********************************************************************************
   * backpropagate: Ber�knar fel/avvikelser i angivet utg�ngslager via angivna
   *                referensv�rden fr�n tr�ningsdatan. OBS! Denna medlemsfunktion
   *                �r avsedd enbart f�r utg�ngslager, se den alternativa
   *                medlemsfunktionen med samma namn f�r dolda lager.
   * 
   *                - reference: Referens till vektor inneh�llande referensv�rden.
   ********************************************************************************/
   void backpropagate(const std::vector<double>& reference)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         const auto dev = reference[i] - this->output[i];
         this->error[i] = dev * delta_relu(this->output[i]);
      }

      return;
   }

   /********************************************************************************
   * backpropagate: Ber�knar fel/avvikelser i angivet dolt lager via parametrar
   *                fr�n n�sta/efterf�ljande lager. OBS! Denna medlemsfunktion
   *                �r avsedd enbart f�r dolda lager, se den alternativa
   *                medlemsfunktionen med samma namn f�r utg�ngslager.
   *
   *                - next_layer: Referens till n�sta/efterf�ljande dense-lager.
   ********************************************************************************/
   void backpropagate(const dense_layer& next_layer)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         auto dev = 0.0;

         for (std::size_t j = 0; j < next_layer.num_nodes(); ++j)
         {
            dev += next_layer.error[j] * next_layer.weights[j][i];
         }

         this->error[i] = dev * this->delta_relu(this->output[i]);
      }

      return;
   }

   /********************************************************************************
   * optimize: Justerar bias och vikter i angivet dense-lager utefter ber�knade
   *           felv�rden samt angiven l�rhastighet. F�r att justera vikterna tas
   *           insignalerna i �tanke, d� h�gre insignal inneb�r att en given vikt
   *           haft h�gre p�verkan och d�rmed h�gre bidrag vid eventuellt fel.
   *           D�rmed justeras vikter med h�ga insignaler i h�gre grad f�r att
   *           minska aktuella felv�rden.
   * 
   *           - input        : Referens till vektor inneh�llande insignaler, 
   *                            som anv�nds f�r att justera vikterna.
   *           - learning_rate: Indikerar hur h�g andel av aktuell fel som
   *                            bias och vikter ska justeras.
   ********************************************************************************/
   void optimize(const std::vector<double>& input,
                 const double learning_rate)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         this->bias[i] += this->error[i] * learning_rate;

         for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
         {
            this->weights[i][j] += this->error[i] * learning_rate * input[j];
         }
      }

      return;
   }

private:
   /********************************************************************************
   * get_random: Returnerar ett randomiserat flyttal mellan 0.0 - 1.0.
   ********************************************************************************/
   static inline double get_random(void)
   {
      return static_cast<double>(std::rand()) / RAND_MAX;
   }

   /********************************************************************************
   * relu: Implementerar en ReLU-funktion, d�r utsignalen y s�tts till angiven
   *       summa sum om denna summa �verstiger 0, annars s�tts utsignalen till 0.
   *       Utsignalen y returneras efter ber�kning. Om summan �verstiger 0 s� �r 
   *       aktuell nod aktiverad, annars �r den inaktiverad:
   * 
   *       sum > 0  => y = sum (noden aktiverad)
   *       sum <= 0 => y = 0   (noden inaktiverad)
   * 
   *       - sum: Summan av nodens bias samt  indata (insignaler * vikter).
   ********************************************************************************/
   static inline double relu(const double sum)
   {
      return sum > 0.0 ? sum : 0.0;
   }

   /********************************************************************************
   * delta_relu: Implementerar en funktion f�r att ber�kna derivatan av 
   *             ReLU-funktionen, d�r utsignalen dy s�tts till 1.0 om angiven nod
   *             �r aktiverad, vilket kontrolleras via ing�ende argument output,
   *             som utg�rs av nodens aktuella utsignal. Annars returneras 0.0.
   *             Via detta v�rde r�knas endast fel/avvikelser p� aktiverade noder,
   *             vilket ocks� medf�r att enbart aktiverade noder justeras vid
   *             optimering. Utsignal dy returneras vid �terhopp enligt nedan:
   *
   *             output > 0  => dy = 1 (noden aktiverad, fel ber�knas)
   *             output <= 0 => dy = 0 (noden inaktiverad, fel ber�knas inte)
   ********************************************************************************/
   static inline double delta_relu(const double output)
   {
      return output > 0.0 ? 1.0 : 0.0;
   }

   /********************************************************************************
   * get_rounded: Kontrollerar angivet flyttal och returnerar noll ifall detta
   *              ligger inom angivet intervall [-threshold, threshold]. 
   *              I praktiken g�ller d�rmed att flyttalet j�mf�rs med 
   *              absolutbeloppet av angivet tr�skelv�rde. Annars om flyttalet
   *              i fr�ga inte ligger inom intervallet returneras detta tal.
   * 
   *              - number   : Flyttalet som ska kontrolleras.
   *              - threshold: Tr�skelv�rdet som anv�nds f�r j�mf�relse.
   ********************************************************************************/
   static double get_rounded(const double number,
                             const double threshold = 0.001)
   {
      if (number > -threshold && number < threshold)
      {
         return 0;
      }
      else
      {
         return number;
      }
   }
};

#endif /* DENSE_LAYER_HPP_ */