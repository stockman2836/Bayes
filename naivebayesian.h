#ifndef NAIVEBAYESIAN_H_
#define NAIVEBAYESIAN_H_

#include <vector>

#include "bayesian.h"

namespace machinelearning {
namespace baysian {

class NaiveBayesian : public Bayesian {
 public:
  NaiveBayesian(char *);
  // inicializovať všetky potrebné informácie z tréningových dát
  void Train(char *);
  std::vector<int> Predict(char *, bool);
  // vypočítajte pravdepodobnosť každej voľby
  // a vyberte tú najväčšiu ako našu predpoveď
 private:
  std::vector<std::vector<long double> > probabilityTable;
};

}  // prostredie baysian
}  // prostredie machinelearning
#endif
