#include <vector>

#include "caffe/layers/im2col_layer.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void Im2colLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  const int input_num_dims = bottom[0]->shape().size();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_dim = channel_axis_ + 1;
  num_spatial_axes_ = input_num_dims - first_spatial_dim;
  vector<int> dim_blob_shape(1, num_spatial_axes_);
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
}

template <typename Dtype>
void Im2colLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  const int* kernel_shape_data = kernel_shape_.cpu_data();
  const int* stride_data = stride_.cpu_data();
  const int* pad_data = pad_.cpu_data();
  const int* dilation_data = dilation_.cpu_data();
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape[channel_axis_] *= kernel_shape_data[i];
    const int input_dim = bottom[0]->shape(channel_axis_ + i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    top_shape[channel_axis_ + i + 1] = output_dim;
  }
  top[0]->Reshape(top_shape);
  num_ = bottom[0]->count(0, channel_axis_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);

  channels_ = bottom[0]->shape(channel_axis_);
}

template <typename Dtype>
void Im2colLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < num_; ++n) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(bottom_data + n * bottom_dim_, channels_,
          bottom[0]->shape(channel_axis_ + 1),
          bottom[0]->shape(channel_axis_ + 2),
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1],
          top_data + n * top_dim_);
    } else {
      im2col_nd_cpu(bottom_data + n * bottom_dim_, num_spatial_axes_,
          bottom[0]->shape().data() + channel_axis_,
          top[0]->shape().data() + channel_axis_,
          kernel_shape_.cpu_data(), pad_.cpu_data(), stride_.cpu_data(),
          dilation_.cpu_data(), top_data + n * top_dim_);
    }
  }
}

template <typename Dtype>
void Im2colLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int n = 0; n < num_; ++n) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(top_diff + n * top_dim_, channels_,
          bottom[0]->shape(channel_axis_ + 1),
          bottom[0]->shape(channel_axis_ + 2),
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1],
          bottom_diff + n * bottom_dim_);
    } else {
      col2im_nd_cpu(top_diff + n * top_dim_, num_spatial_axes_,
          bottom[0]->shape().data() + channel_axis_,
          top[0]->shape().data() + channel_axis_,
          kernel_shape_.cpu_data(), pad_.cpu_data(), stride_.cpu_data(),
          dilation_.cpu_data(), bottom_diff + n * bottom_dim_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Im2colLayer);
#endif

INSTANTIATE_CLASS(Im2colLayer);
REGISTER_LAYER_CLASS(Im2col);

}  // namespace caffe
