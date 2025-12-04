#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "naivebayesian.h"

using namespace std;

// Hlavný vstupný bod programu pre klasifikáciu s využitím Bajzových sietí
int main(int argc, char **argv) {
  int method = 0;
  char *train;
  char *test;
  char *cfg;
  clock_t begin; // Začiatočný čas vykonávania
  clock_t end; // Koncový čas vykonávania
  double time_spent; // Vypočítaný čas vykonávania
  vector<int> prediction_result; // Výsledky predikcie

  // Spracovanie vstupných argumentov
  if (argc >= 5) {
    method = atoi(argv[4]); // Konverzia metódy zo vstupného argumentu
    train = argv[1]; // Cesta k súboru s tréningovými údajmi
    test = argv[2]; // Cesta k súboru s testovacími údajmi
    cfg = argv[3]; // Cesta k konfiguračnému súboru
  } else if (argc == 4) {
    train = argv[1];
    test = argv[2];
    cfg = argv[3];
    std::cout << " use default naiveBayesian method" << std::endl; // Použitie predvolenej metódy naiveBayesian
  } else {
    std::cout << " You need to provide training data, test data, and "
                 "configuration for prediction. Please read README"
              << std::endl; // Informácia o nutnosti poskytnúť potrebné dáta
  }
  // Začiatok merania času
  begin = clock();
  // Rozhodnutie o metóde na základe vstupného argumentu
  if (method == 0) {
    machinelearning::baysian::NaiveBayesian naive(cfg);
    naive.Train(train); // Trénovanie modelu
    prediction_result = naive.Predict(test, 1);
    // Zavolajte funkciu na predpovedanie
    // Zadajte druhý argument "1" na označenie, pre ktorý poskytneme odpoveď/pravdu
    // túto predpoveď. Zadajte druhý argument "0", aby ste urobili skutočnú predpoveď
  };

  // Koniec merania času a výpis stráveného času
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  std::cout << "Time spent " << time_spent << " seconds " << std::endl; // Vypíše strávený čas
  return 0;
}
