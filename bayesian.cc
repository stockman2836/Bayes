#include "bayesian.h"

#include <fstream>
#include <iostream>

namespace machinelearning {
namespace baysian {

void Bayesian::ParseConfiguration(char *cfg_file) {
  std::ifstream configure;
  configure.open(cfg_file); //otvára konfiguračný súbor
  if (!configure) {
    std::cout << "Can't open configuration file!" << std::endl; // vypíše chybu, ak sa súbor nedá otvoriť
    return;
  }

  configure >> num_train_instances_ >> num_test_instances_ >> num_attributes_;
  // prečíta počet inštancií a atribútov školenia

  is_discrete_.resize(num_attributes_);
  // toto pole uchováva informácie o každom atribúte nepretržité alebo nie
  for (int i = 0; i < num_attributes_; ++i) configure >> is_discrete_[i];
  //  prečíta informácie o nepretržitom alebo nie

  num_class_for_each_attribute_.resize(num_attributes_ + 1);
  // toto pole uchováva počet tried každého atribútu

  for (int i = 0; i <= num_attributes_; ++i) {  // prečíta počet tried
    configure >> num_class_for_each_attribute_[i];
    if (i != num_attributes_ &&  is_discrete_[i])  // nastavte num_class_for_each_attribute_ ako 2 pre
                          // priebežné údaje
      num_class_for_each_attribute_[i] = 2;
  }

  num_output_class_ = num_class_for_each_attribute_[num_attributes_];
  output_class_cnt_.resize(num_output_class_, 0);

  configure.close(); // zatvára konfiguračný súbor po načítaní všetkých údajov
}

}  // prostredie baysian
}  // prostredie machinelearning
