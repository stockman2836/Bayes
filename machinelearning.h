#ifndef ML_H_
#define ML_H_

#include <vector>

namespace machinelearning {

class MachineLearning {
 public:
  virtual void Train(char *) = 0;
  virtual std::vector<int> Predict(char *, bool) = 0;
  void Accuracy(std::vector<int> &, std::vector<int> &) const;

 protected:
  virtual void ParseConfiguration(char *) = 0;
  int num_train_instances_;  // uložiť počet inštancií školenia
  int num_test_instances_;   // uložiť počet testovacích inštancií
};

}  // prostredie machinelearning
#endif
