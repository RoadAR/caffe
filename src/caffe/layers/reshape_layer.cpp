#include <vector>

#include "caffe/layers/reshape_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  inferred_axis_ = -1;
  copy_axes_.clear();
  const BlobShape& top_blob_shape = this->layer_param_.reshape_param().shape();
  const int top_num_axes = top_blob_shape.dim_size();
  constant_count_ = 1;
  for (int i = 0; i < top_num_axes; ++i) {
    const int top_dim = top_blob_shape.dim(i);
    if (top_dim == 0) {
      copy_axes_.push_back(i);
    } else if (top_dim == -1) {
      inferred_axis_ = i;
    } else {
      constant_count_ *= top_dim;
    }
  }
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int input_start_axis = this->layer_param_.reshape_param().axis();
  const int start_axis = (input_start_axis >= 0) ? input_start_axis :
      bottom[0]->num_axes() + input_start_axis + 1;
  const int num_axes = this->layer_param_.reshape_param().num_axes();
  const int end_axis =
      (num_axes == -1) ? bottom[0]->num_axes() : (start_axis + num_axes);
  const int num_axes_replaced = end_axis - start_axis;
  const int num_axes_retained = bottom[0]->num_axes() - num_axes_replaced;
  const BlobShape& top_blob_shape = this->layer_param_.reshape_param().shape();
  const int num_new_axes = top_blob_shape.dim_size();
  vector<int> top_shape(num_axes_retained + num_new_axes);
  int top_shape_index = 0;
  for (int i = 0; i < start_axis; ++i) {
    top_shape[top_shape_index++] = bottom[0]->shape(i);
  }
  for (int i = 0; i < num_new_axes; ++i) {
    top_shape[top_shape_index++] = top_blob_shape.dim(i);
  }
  for (int i = end_axis; i < bottom[0]->num_axes(); ++i) {
    top_shape[top_shape_index++] = bottom[0]->shape(i);
  }
  for (int i = 0; i < copy_axes_.size(); ++i) {
    const int copy_axis_index = copy_axes_[i];
    top_shape[start_axis + copy_axis_index] =
        bottom[0]->shape(start_axis + copy_axis_index);
  }
  if (inferred_axis_ >= 0) {
    // A -1 dim was specified; infer the correct dimension by computing the
    // product of the other dimensions.
    int explicit_count = constant_count_;
    explicit_count *= bottom[0]->count(0, start_axis);
    explicit_count *= bottom[0]->count(end_axis);
    for (int i = 0; i < copy_axes_.size(); ++i) {
      const int copy_axis_index = copy_axes_[i];
      explicit_count *= top_shape[start_axis + copy_axis_index];
    }
    const int inferred_dim = bottom[0]->count() / explicit_count;
    top_shape[start_axis + inferred_axis_] = inferred_dim;
  }
  top[0]->Reshape(top_shape);
  top[0]->ShareData(*bottom[0]);
  top[0]->ShareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(Reshape);

}  // namespace caffe
