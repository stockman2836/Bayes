#ifndef BAYESIAN_H_
#define BAYESIAN_H_

#include "machinelearning.h"

namespace machinelearning {
namespace baysian {

class Bayesian : public MachineLearning {
 protected:
  void ParseConfiguration(char *);
  std::vector<double> output_class_cnt_;
  // toto pole ukladá celkový počet
  // trieda každého rozhodnutia v tréningových údajoch
  std::vector<int> is_discrete_;
  // toto pole uchováva informácie o každom atribúte
  // je nepretržitý alebo nie
  std::vector<int> num_class_for_each_attribute_;
  // toto pole uchováva počet tried každého atribútu
  int num_attributes_;                 // uložiť počet atribútov
  int num_output_class_;               // počet výstupných tried
};

}  // prostredie baysian
}  // prostredie machinelearning
#endif
