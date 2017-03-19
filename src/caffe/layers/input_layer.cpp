#include <vector>

#include "caffe/layers/input_layer.hpp"

namespace caffe {

template <typename Dtype>
void InputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_top = top.size();
  const InputParameter& param = this->layer_param_.input_param();
  const int num_shape = param.shape_size();
  if (num_shape > 0) {
    for (int i = 0; i < num_top; ++i) {
      const int shape_index = (param.shape_size() == 1) ? 0 : i;
      top[i]->Reshape(param.shape(shape_index));
    }
  }
}

INSTANTIATE_CLASS(InputLayer);
REGISTER_LAYER_CLASS(Input);

}  // namespace caffe
